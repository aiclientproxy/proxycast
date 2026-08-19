use agent_protocol::hook::HookRunSummary;
use agent_protocol::provider_trace::{ProviderTraceEvent, ProviderTraceStage};
use agent_protocol::turn_context::TurnOutputSchemaRuntime;
use agent_protocol::{
    CollabAgentOperation, ItemId, ItemStatus, SessionId, ThreadId, ThreadItem, ThreadItemPayload,
    ToolArgument, ToolOutput, TurnId,
};
use lime_core::database::dao::agent_timeline::AgentThreadTurn;
use model_provider::current_client::{ModelRerouteReason, ModelVerification};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;
use tool_runtime::tool_lifecycle::{ToolLifecycleEvent, ToolLifecyclePhase};
use tool_runtime::tool_result_projection::NormalizedToolOutput;

use crate::session_execution_runtime::{
    SessionExecutionRuntimeCostState, SessionExecutionRuntimeLimitEvent,
    SessionExecutionRuntimeLimitState, SessionExecutionRuntimeRoutingDecision,
    SessionExecutionRuntimeTaskProfile,
};

mod protocol_agent_control;
pub(crate) use protocol_agent_control::CanonicalSubAgentActivity;

pub type AgentActionRequiredScope = agent_protocol::action_required::ActionRequiredScope;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentToolImage {
    pub src: String,
    #[serde(rename = "mimeType", skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentToolResult {
    pub success: bool,
    pub output: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(
        default,
        rename = "structuredContent",
        skip_serializing_if = "Option::is_none"
    )]
    pub structured_content: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<AgentToolImage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentToolProgressPayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentArtifactSignal {
    pub artifact_id: String,
    pub file_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentTokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_input_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(
        default,
        rename = "codex_rollout_budget_units",
        skip_serializing_if = "Option::is_none"
    )]
    pub codex_rollout_budget_units: Option<serde_json::Number>,
}

pub type AgentContextTraceStep = agent_protocol::context_trace::ContextTraceStep;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AgentContextBudget {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub used_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remaining_tokens: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentMissingContextFact {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub kind: String,
    pub label: String,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentRetrievalRef {
    pub source_id: String,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentTeamMemoryRef {
    pub key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo_scope: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AgentTurnContextSummary {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_budget: Option<AgentContextBudget>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub missing_context: Vec<AgentMissingContextFact>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub retrieval_refs: Vec<AgentRetrievalRef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub team_memory_refs: Vec<AgentTeamMemoryRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRuntimeStatus {
    pub phase: String,
    pub title: String,
    pub detail: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub checkpoints: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, Value>>,
}

pub fn build_diagnostics_runtime_status_metadata() -> HashMap<String, Value> {
    agent_runtime::runtime_timeline::build_diagnostics_runtime_status_metadata()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub role: String,
    pub content: Vec<AgentMessageContent>,
    pub timestamp: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<AgentTokenUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AgentMessageContent {
    #[serde(rename = "text")]
    Text { text: String },

    #[serde(rename = "thinking")]
    Thinking { text: String },

    #[serde(rename = "tool_request")]
    ToolRequest {
        id: String,
        tool_name: String,
        arguments: Value,
    },

    #[serde(rename = "tool_response")]
    ToolResponse {
        id: String,
        success: bool,
        output: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        #[serde(
            default,
            rename = "structuredContent",
            skip_serializing_if = "Option::is_none"
        )]
        structured_content: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        images: Option<Vec<AgentToolImage>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        metadata: Option<HashMap<String, Value>>,
    },

    #[serde(rename = "action_required")]
    ActionRequired {
        id: String,
        action_type: String,
        data: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        scope: Option<AgentActionRequiredScope>,
    },

    #[serde(rename = "image")]
    Image { mime_type: String, data: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextDeltaBatchBoundary {
    Newline,
    Backlog,
    Final,
    Provider,
}

pub type AgentProviderTraceStage = ProviderTraceStage;
pub type AgentProviderTraceEvent = ProviderTraceEvent;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentMessagePhase {
    Commentary,
    FinalAnswer,
}

impl AgentMessagePhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Commentary => "commentary",
            Self::FinalAnswer => "final_answer",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum GuardianReviewStatus {
    InProgress,
    Approved,
    Denied,
    TimedOut,
    Aborted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GuardianRiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GuardianUserAuthorization {
    Unknown,
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AgentEvent {
    /// Internal trace evidence. App Server consumes this before RuntimeEvent
    /// projection, so it never becomes a public ThreadItem or GUI event.
    #[serde(rename = "code_cell_trace")]
    CodeCellTrace {
        event: tool_runtime::tool_lifecycle::CodeCellTraceEvent,
    },

    #[serde(rename = "thread_started")]
    ThreadStarted { thread_id: String },

    #[serde(rename = "turn_started")]
    TurnStarted { turn: AgentThreadTurn },

    #[serde(rename = "item_started")]
    ItemStarted { item: ThreadItem },

    #[serde(rename = "item_updated")]
    ItemUpdated { item: ThreadItem },

    #[serde(rename = "item_completed")]
    ItemCompleted { item: ThreadItem },

    #[serde(rename = "turn_completed")]
    TurnCompleted { turn: AgentThreadTurn },

    #[serde(rename = "turn_failed")]
    TurnFailed { turn: AgentThreadTurn },

    #[serde(rename = "text_start")]
    TextStart {
        #[serde(rename = "itemId")]
        item_id: String,
    },

    #[serde(rename = "text_delta")]
    TextDelta {
        #[serde(rename = "itemId")]
        item_id: String,
        text: String,
    },

    #[serde(rename = "text_end")]
    TextEnd {
        #[serde(rename = "itemId")]
        item_id: String,
        phase: AgentMessagePhase,
    },

    #[serde(rename = "text_delta_batch")]
    TextDeltaBatch {
        text: String,
        chunks: Vec<String>,
        boundary: TextDeltaBatchBoundary,
    },

    #[serde(rename = "reasoning_start")]
    ReasoningStart {
        #[serde(rename = "itemId")]
        item_id: String,
    },

    #[serde(rename = "reasoning_summary_delta")]
    ReasoningSummaryDelta {
        #[serde(rename = "itemId")]
        item_id: String,
        text: String,
        #[serde(rename = "summaryIndex")]
        summary_index: i64,
    },

    #[serde(rename = "reasoning_summary_part_added")]
    ReasoningSummaryPartAdded {
        #[serde(rename = "itemId")]
        item_id: String,
        #[serde(rename = "summaryIndex")]
        summary_index: i64,
    },

    #[serde(rename = "reasoning_content_delta")]
    ReasoningContentDelta {
        #[serde(rename = "itemId")]
        item_id: String,
        text: String,
        #[serde(rename = "contentIndex")]
        content_index: i64,
    },

    #[serde(rename = "reasoning_end")]
    ReasoningEnd {
        #[serde(rename = "itemId")]
        item_id: String,
    },

    #[serde(rename = "tool_progress")]
    ToolProgress {
        tool_id: String,
        progress: AgentToolProgressPayload,
    },

    #[serde(rename = "tool_output_delta")]
    ToolOutputDelta {
        tool_id: String,
        delta: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        output_kind: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        metadata: Option<HashMap<String, Value>>,
    },

    #[serde(rename = "tool_input_delta")]
    ToolInputDelta {
        tool_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_name: Option<String>,
        delta: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        accumulated_arguments: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider: Option<String>,
    },

    #[serde(rename = "hook_started")]
    HookStarted {
        turn_id: Option<String>,
        run: HookRunSummary,
    },

    #[serde(rename = "hook_completed")]
    HookCompleted {
        turn_id: Option<String>,
        run: HookRunSummary,
    },

    #[serde(rename = "artifact_snapshot")]
    ArtifactSnapshot { artifact: AgentArtifactSignal },

    #[serde(rename = "action_required")]
    ActionRequired {
        request_id: String,
        action_type: String,
        data: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        scope: Option<AgentActionRequiredScope>,
    },

    #[serde(rename = "action_resolved")]
    ActionResolved {
        request_id: String,
        action_type: String,
        data: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        scope: Option<AgentActionRequiredScope>,
    },

    #[serde(rename = "dynamic_tool_call_requested")]
    DynamicToolCallRequested {
        call_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        namespace: Option<String>,
        tool: String,
        arguments: Value,
    },

    #[serde(rename = "turn_context")]
    TurnContext {
        session_id: String,
        thread_id: String,
        turn_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        execution_strategy: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        output_schema_runtime: Option<TurnOutputSchemaRuntime>,
        #[serde(skip_serializing_if = "Option::is_none")]
        context_summary: Option<AgentTurnContextSummary>,
        #[serde(skip_serializing_if = "Option::is_none")]
        approval_policy: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        sandbox_policy: Option<String>,
    },

    #[serde(rename = "model_change")]
    ModelChange { model: String, mode: String },

    #[serde(rename = "server_model")]
    ServerModel { model: String },

    #[serde(rename = "model_reroute")]
    ModelReroute {
        from_model: String,
        to_model: String,
        reason: ModelRerouteReason,
    },

    #[serde(rename = "model_verification")]
    ModelVerification {
        verifications: Vec<ModelVerification>,
    },

    #[serde(rename = "turn_moderation_metadata")]
    TurnModerationMetadata { metadata: Value },

    #[serde(rename = "guardian_review_started")]
    GuardianReviewStarted {
        review_id: String,
        #[serde(rename = "targetItemId", skip_serializing_if = "Option::is_none")]
        target_item_id: Option<String>,
        #[serde(rename = "turnId")]
        turn_id: String,
        action: Value,
        #[serde(rename = "startedAtMs")]
        started_at_ms: i64,
    },

    #[serde(rename = "guardian_review_completed")]
    GuardianReviewCompleted {
        review_id: String,
        #[serde(rename = "targetItemId", skip_serializing_if = "Option::is_none")]
        target_item_id: Option<String>,
        #[serde(rename = "turnId")]
        turn_id: String,
        status: GuardianReviewStatus,
        #[serde(rename = "riskLevel", skip_serializing_if = "Option::is_none")]
        risk_level: Option<GuardianRiskLevel>,
        #[serde(rename = "userAuthorization", skip_serializing_if = "Option::is_none")]
        user_authorization: Option<GuardianUserAuthorization>,
        #[serde(skip_serializing_if = "Option::is_none")]
        rationale: Option<String>,
        #[serde(rename = "decisionSource")]
        decision_source: String,
        action: Value,
        #[serde(rename = "startedAtMs")]
        started_at_ms: i64,
        #[serde(rename = "completedAtMs")]
        completed_at_ms: i64,
    },

    #[serde(rename = "guardian_warning")]
    GuardianWarning { message: String },

    #[serde(rename = "provider_trace")]
    ProviderTrace {
        #[serde(flatten)]
        event: AgentProviderTraceEvent,
    },

    #[serde(rename = "provider_stream_event")]
    ProviderStreamEvent {
        runtime_event_kind: String,
        payload: Value,
    },

    #[serde(rename = "context_trace")]
    ContextTrace { steps: Vec<AgentContextTraceStep> },

    #[serde(rename = "context_compaction_started")]
    ContextCompactionStarted {
        item_id: String,
        trigger: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },

    #[serde(rename = "context_compaction_completed")]
    ContextCompactionCompleted {
        item_id: String,
        trigger: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },

    #[serde(rename = "runtime_status")]
    RuntimeStatus { status: AgentRuntimeStatus },

    #[serde(rename = "task_profile_resolved")]
    TaskProfileResolved {
        task_profile: SessionExecutionRuntimeTaskProfile,
    },

    #[serde(rename = "candidate_set_resolved")]
    CandidateSetResolved {
        routing_decision: SessionExecutionRuntimeRoutingDecision,
    },

    #[serde(rename = "routing_decision_made")]
    RoutingDecisionMade {
        routing_decision: SessionExecutionRuntimeRoutingDecision,
    },

    #[serde(rename = "routing_fallback_applied")]
    RoutingFallbackApplied {
        routing_decision: SessionExecutionRuntimeRoutingDecision,
    },

    #[serde(rename = "routing_not_possible")]
    RoutingNotPossible {
        routing_decision: SessionExecutionRuntimeRoutingDecision,
    },

    #[serde(rename = "limit_state_updated")]
    LimitStateUpdated {
        limit_state: SessionExecutionRuntimeLimitState,
    },

    #[serde(rename = "single_candidate_only")]
    SingleCandidateOnly {
        limit_state: SessionExecutionRuntimeLimitState,
    },

    #[serde(rename = "single_candidate_capability_gap")]
    SingleCandidateCapabilityGap {
        limit_state: SessionExecutionRuntimeLimitState,
    },

    #[serde(rename = "cost_estimated")]
    CostEstimated {
        cost_state: SessionExecutionRuntimeCostState,
    },

    #[serde(rename = "cost_recorded")]
    CostRecorded {
        cost_state: SessionExecutionRuntimeCostState,
    },

    #[serde(rename = "rate_limit_hit")]
    RateLimitHit {
        limit_event: SessionExecutionRuntimeLimitEvent,
    },

    #[serde(rename = "quota_low")]
    QuotaLow {
        limit_event: SessionExecutionRuntimeLimitEvent,
    },

    #[serde(rename = "quota_blocked")]
    QuotaBlocked {
        limit_event: SessionExecutionRuntimeLimitEvent,
    },

    #[serde(rename = "provider_usage")]
    ProviderUsage {
        attempt: u32,
        usage: AgentTokenUsage,
    },

    #[serde(rename = "rollout_budget_reminder")]
    RolloutBudgetReminder {
        #[serde(rename = "remainingTokens")]
        remaining_tokens: i64,
        #[serde(rename = "reminderIndex")]
        reminder_index: usize,
        #[serde(rename = "windowId")]
        window_id: String,
        #[serde(rename = "durableEventId")]
        durable_event_id: String,
        text: String,
    },

    #[serde(rename = "provider_step")]
    ProviderStep {
        attempt: u32,
        completed: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        finish_reason: Option<String>,
        text_output_chars: u64,
        reasoning_output_chars: u64,
        tool_call_count: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        usage: Option<AgentTokenUsage>,
    },

    #[serde(rename = "done")]
    Done {
        #[serde(skip_serializing_if = "Option::is_none")]
        usage: Option<AgentTokenUsage>,
    },

    #[serde(rename = "error")]
    Error { message: String },

    #[serde(rename = "warning")]
    Warning {
        #[serde(skip_serializing_if = "Option::is_none")]
        code: Option<String>,
        message: String,
    },

    #[serde(rename = "message")]
    Message { message: AgentMessage },
}

#[derive(Clone)]
pub(crate) struct ToolItemLifecycleContext {
    pub session_id: SessionId,
    pub thread_id: ThreadId,
    pub sequence: u64,
    pub ordinal: u64,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
}

pub(crate) fn canonical_tool_item_event(
    event: ToolLifecycleEvent,
    context: ToolItemLifecycleContext,
) -> Option<AgentEvent> {
    let ToolLifecycleEvent {
        turn_id,
        call_id,
        tool_name,
        arguments,
        provider_metadata,
        environments,
        phase,
        output,
    } = event;
    if matches!(phase, ToolLifecyclePhase::Completed) && output.is_none() {
        return None;
    }
    if tool_name == tool_runtime::unified_exec::WRITE_STDIN_TOOL_NAME
        && matches!(phase, ToolLifecyclePhase::Started)
    {
        return None;
    }

    let raw_arguments = arguments.clone();
    let arguments = canonical_tool_arguments(arguments);
    let mut metadata = serde_json::Map::new();
    if provider_metadata
        .as_object()
        .is_some_and(|metadata| !metadata.is_empty())
    {
        metadata.insert("provider_metadata".to_string(), provider_metadata);
    }
    metadata.insert(
        "environments".to_string(),
        Value::Array(
            environments
                .into_iter()
                .map(|environment| {
                    serde_json::json!({
                        "environmentId": environment.environment_id,
                        "cwd": environment.cwd.to_string_lossy(),
                    })
                })
                .collect(),
        ),
    );

    let mut agent_states = HashMap::new();
    let (mut status, output) = match output {
        Some(output) => {
            let NormalizedToolOutput {
                success,
                text,
                structured_content,
                error,
                duration_ms,
                truncation,
                sidecar_reference,
                metadata: output_metadata,
                agent_control_projection_facts: _,
                agent_control_state_facts,
            } = output;
            agent_states.extend(
                agent_control_state_facts
                    .into_iter()
                    .map(|fact| (fact.target_thread_id, fact.state)),
            );
            let aborted = output_metadata
                .get(tool_runtime::tool_result_projection::TOOL_OUTCOME_METADATA_KEY)
                .and_then(Value::as_str)
                == Some(tool_runtime::tool_result_projection::TOOL_OUTCOME_ABORTED);
            metadata.extend(output_metadata);
            metadata.insert("success".to_string(), Value::Bool(success));
            metadata.insert("duration_ms".to_string(), Value::from(duration_ms));
            if let Some(truncation) = truncation.as_ref() {
                metadata.insert(
                    "truncation".to_string(),
                    serde_json::to_value(truncation).unwrap_or(Value::Null),
                );
            }
            if let Some(reference) = sidecar_reference.as_ref() {
                metadata.insert(
                    "sidecar_reference".to_string(),
                    serde_json::to_value(reference).unwrap_or(Value::Null),
                );
            }
            (
                if aborted {
                    ItemStatus::Cancelled
                } else if success {
                    ItemStatus::Completed
                } else {
                    ItemStatus::Failed
                },
                Some(ToolOutput {
                    text: Some(text),
                    structured_content,
                    error,
                    duration_ms: Some(duration_ms),
                    truncated: truncation.is_some(),
                    output_ref: sidecar_reference.map(|reference| reference.reference),
                }),
            )
        }
        None => (ItemStatus::InProgress, None),
    };
    let command_call_id = metadata
        .get("exec_command_call_id")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let command_running = metadata.get("session_id").is_some_and(Value::is_number)
        && metadata.get("exit_code").is_none_or(Value::is_null);
    let command_exit_code = metadata
        .get("exit_code")
        .and_then(Value::as_i64)
        .and_then(|value| i32::try_from(value).ok());
    let command_tool = tool_name == tool_runtime::unified_exec::EXEC_COMMAND_TOOL_NAME
        || tool_name == tool_runtime::unified_exec::WRITE_STDIN_TOOL_NAME;
    if command_running {
        status = ItemStatus::InProgress;
    }
    if tool_name == tool_runtime::agent_control::WAIT_AGENT_TOOL_NAME
        && agent_states.values().any(|state| {
            matches!(
                state.status,
                agent_protocol::CollabAgentStatus::Errored
                    | agent_protocol::CollabAgentStatus::NotFound
            )
        })
    {
        status = ItemStatus::Failed;
    }
    let payload = if command_tool {
        let command = metadata
            .get("command")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .or_else(|| {
                raw_arguments
                    .get("cmd")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
            })
            .unwrap_or_else(|| "command".to_string());
        let cwd = metadata
            .get("cwd")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .or_else(|| {
                raw_arguments
                    .get("workdir")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
            });
        let command_output = metadata
            .get("command_output")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .filter(|value| !value.is_empty());
        ThreadItemPayload::Command {
            command,
            cwd,
            output: command_output,
            exit_code: command_exit_code,
        }
    } else if tool_name == tool_runtime::agent_control::WAIT_AGENT_TOOL_NAME {
        ThreadItemPayload::CollabAgentToolCall {
            call_id: call_id.clone(),
            operation: CollabAgentOperation::Wait,
            target_thread_id: None,
            message: None,
            output,
            agent_states,
        }
    } else {
        ThreadItemPayload::Tool {
            call_id: call_id.clone(),
            name: tool_name,
            arguments,
            output,
        }
    };
    let item_id = command_call_id.unwrap_or(call_id);
    let item = ThreadItem {
        session_id: context.session_id,
        thread_id: context.thread_id,
        turn_id: TurnId::new(turn_id),
        item_id: ItemId::new(item_id),
        sequence: context.sequence,
        ordinal: context.ordinal,
        created_at_ms: context.created_at_ms,
        updated_at_ms: context.updated_at_ms,
        completed_at_ms: status.is_terminal().then_some(context.updated_at_ms),
        kind: payload.kind(),
        status,
        payload,
        metadata: Value::Object(metadata),
    };
    Some(match (phase, command_running) {
        (ToolLifecyclePhase::Started, _) => AgentEvent::ItemStarted { item },
        (ToolLifecyclePhase::Completed, true) => AgentEvent::ItemUpdated { item },
        (ToolLifecyclePhase::Completed, false) => AgentEvent::ItemCompleted { item },
    })
}

fn canonical_tool_arguments(arguments: Value) -> Vec<ToolArgument> {
    match arguments {
        Value::Object(arguments) => arguments
            .into_iter()
            .map(|(name, value)| ToolArgument {
                name,
                value: compact_tool_argument(value),
            })
            .collect(),
        Value::Array(arguments) => arguments
            .into_iter()
            .enumerate()
            .map(|(index, value)| ToolArgument {
                name: index.to_string(),
                value: compact_tool_argument(value),
            })
            .collect(),
        Value::Null => Vec::new(),
        value => vec![ToolArgument {
            name: "value".to_string(),
            value: compact_tool_argument(value),
        }],
    }
}

fn compact_tool_argument(value: Value) -> String {
    match value {
        Value::String(value) => value,
        value => value.to_string(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentImageAttachment {
    pub data: String,
    #[serde(alias = "mediaType")]
    pub media_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentUserPreferences {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_preference: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_preference: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_search: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_strategy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subagent: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub theme: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_continue: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentUserInputOp {
    pub text: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<AgentImageAttachment>,
    pub preferences: AgentUserPreferences,
    pub session_id: String,
    pub workspace_id: String,
    pub event_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_if_busy: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queued_turn_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
#[allow(clippy::large_enum_variant)]
pub enum AgentOp {
    UserInput(AgentUserInputOp),
    Interrupt {
        session_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        turn_id: Option<String>,
    },
    Retry {
        session_id: String,
        turn_id: String,
    },
    ConfigUpdate {
        session_id: String,
        key: String,
        value: Value,
    },
    Shutdown {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        session_id: Option<String>,
    },
}

#[cfg(test)]
#[path = "protocol_agent_control_tests.rs"]
mod agent_control_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tool_runtime::tool_lifecycle::{ToolLifecycleEvent, ToolLifecyclePhase};
    use tool_runtime::tool_result_projection::NormalizedToolOutput;

    fn tool_item_context() -> ToolItemLifecycleContext {
        ToolItemLifecycleContext {
            session_id: SessionId::new("session-1"),
            thread_id: ThreadId::new("thread-1"),
            sequence: 1,
            ordinal: 1,
            created_at_ms: 1,
            updated_at_ms: 2,
        }
    }

    fn unified_exec_output(running: bool) -> NormalizedToolOutput {
        let mut metadata = HashMap::from([
            (
                "exec_command_call_id".to_string(),
                serde_json::json!("exec-call"),
            ),
            ("command".to_string(), serde_json::json!("printf done")),
            ("cwd".to_string(), serde_json::json!("/tmp/project")),
            (
                "command_output".to_string(),
                serde_json::json!(if running { "" } else { "done" }),
            ),
            (
                "exit_code".to_string(),
                if running {
                    Value::Null
                } else {
                    serde_json::json!(0)
                },
            ),
        ]);
        if running {
            metadata.insert("session_id".to_string(), serde_json::json!(1001));
        }
        NormalizedToolOutput {
            success: true,
            text: "{}".to_string(),
            structured_content: None,
            error: None,
            duration_ms: 10,
            truncation: None,
            sidecar_reference: None,
            metadata,
            agent_control_projection_facts: Vec::new(),
            agent_control_state_facts: Vec::new(),
        }
    }

    #[test]
    fn aborted_tool_output_projects_cancelled_item_status() {
        let event = canonical_tool_item_event(
            ToolLifecycleEvent {
                turn_id: "turn-aborted".to_string(),
                call_id: "call-aborted".to_string(),
                tool_name: "sleep".to_string(),
                arguments: serde_json::json!({}),
                provider_metadata: Value::Null,
                environments: Vec::new(),
                phase: ToolLifecyclePhase::Completed,
                output: Some(NormalizedToolOutput {
                    success: false,
                    text: "Tool execution aborted".to_string(),
                    structured_content: None,
                    error: None,
                    duration_ms: 1,
                    truncation: None,
                    sidecar_reference: None,
                    metadata: HashMap::from([(
                        tool_runtime::tool_result_projection::TOOL_OUTCOME_METADATA_KEY.to_string(),
                        Value::String(
                            tool_runtime::tool_result_projection::TOOL_OUTCOME_ABORTED.to_string(),
                        ),
                    )]),
                    agent_control_projection_facts: Vec::new(),
                    agent_control_state_facts: Vec::new(),
                }),
            },
            tool_item_context(),
        )
        .expect("aborted tool lifecycle event");

        let AgentEvent::ItemCompleted { item } = event else {
            panic!("aborted tool should produce one terminal item");
        };
        assert_eq!(item.status, ItemStatus::Cancelled);
    }

    #[test]
    fn exec_command_projects_canonical_command_lifecycle() {
        let started = canonical_tool_item_event(
            ToolLifecycleEvent {
                turn_id: "turn-1".to_string(),
                call_id: "exec-call".to_string(),
                tool_name: "exec_command".to_string(),
                arguments: serde_json::json!({ "cmd": "printf done" }),
                provider_metadata: Value::Null,
                environments: Vec::new(),
                phase: ToolLifecyclePhase::Started,
                output: None,
            },
            tool_item_context(),
        )
        .expect("command started event");
        let AgentEvent::ItemStarted { item } = started else {
            panic!("expected item started");
        };
        assert!(matches!(item.payload, ThreadItemPayload::Command { .. }));

        let updated = canonical_tool_item_event(
            ToolLifecycleEvent {
                turn_id: "turn-1".to_string(),
                call_id: "exec-call".to_string(),
                tool_name: "exec_command".to_string(),
                arguments: serde_json::json!({ "cmd": "printf done" }),
                provider_metadata: Value::Null,
                environments: Vec::new(),
                phase: ToolLifecyclePhase::Completed,
                output: Some(unified_exec_output(true)),
            },
            tool_item_context(),
        )
        .expect("running command update");
        let AgentEvent::ItemUpdated { item } = updated else {
            panic!("expected item updated");
        };
        assert_eq!(item.item_id.as_str(), "item_exec-call");
        assert_eq!(item.status, ItemStatus::InProgress);
    }

    #[test]
    fn write_stdin_completes_original_command_item_without_second_started_item() {
        let started = canonical_tool_item_event(
            ToolLifecycleEvent {
                turn_id: "turn-1".to_string(),
                call_id: "write-call".to_string(),
                tool_name: "write_stdin".to_string(),
                arguments: serde_json::json!({ "session_id": 1001, "chars": "" }),
                provider_metadata: Value::Null,
                environments: Vec::new(),
                phase: ToolLifecyclePhase::Started,
                output: None,
            },
            tool_item_context(),
        );
        assert!(started.is_none());

        let completed = canonical_tool_item_event(
            ToolLifecycleEvent {
                turn_id: "turn-1".to_string(),
                call_id: "write-call".to_string(),
                tool_name: "write_stdin".to_string(),
                arguments: serde_json::json!({ "session_id": 1001, "chars": "" }),
                provider_metadata: Value::Null,
                environments: Vec::new(),
                phase: ToolLifecyclePhase::Completed,
                output: Some(unified_exec_output(false)),
            },
            tool_item_context(),
        )
        .expect("original command completion");
        let AgentEvent::ItemCompleted { item } = completed else {
            panic!("expected item completed");
        };
        assert_eq!(item.item_id.as_str(), "item_exec-call");
        assert!(matches!(
            item.payload,
            ThreadItemPayload::Command {
                output: Some(ref output),
                exit_code: Some(0),
                ..
            } if output == "done"
        ));
    }

    #[test]
    fn agent_op_user_input_serializes_with_protocol_tag() {
        let value = serde_json::to_value(AgentOp::UserInput(AgentUserInputOp {
            text: "继续处理当前任务".to_string(),
            images: vec![AgentImageAttachment {
                data: "aGVsbG8=".to_string(),
                media_type: "image/png".to_string(),
            }],
            preferences: AgentUserPreferences {
                provider_preference: Some("openai".to_string()),
                model_preference: Some("gpt-5.4".to_string()),
                thinking: Some(true),
                web_search: Some(false),
                search_mode: Some("disabled".to_string()),
                execution_strategy: Some("react".to_string()),
                task: Some(false),
                subagent: Some(false),
                theme: Some("general".to_string()),
                auto_continue: Some(serde_json::json!({
                    "enabled": true,
                    "continuation_length": 3
                })),
            },
            session_id: "session-1".to_string(),
            workspace_id: "workspace-1".to_string(),
            event_name: "agent_stream_session-1".to_string(),
            turn_id: Some("turn-1".to_string()),
            system_prompt: Some("保持简洁".to_string()),
            metadata: Some(Map::from_iter([(
                "theme".to_string(),
                Value::String("general".to_string()),
            )])),
            queue_if_busy: Some(true),
            queued_turn_id: Some("queued-1".to_string()),
        }))
        .expect("serialize agent op");

        assert_eq!(value["type"], "user_input");
        assert_eq!(value["preferences"]["provider_preference"], "openai");
        assert_eq!(value["session_id"], "session-1");
        assert_eq!(value["queue_if_busy"], true);
    }

    #[test]
    fn agent_op_interrupt_deserializes_from_snake_case_tag() {
        let op: AgentOp = serde_json::from_value(serde_json::json!({
            "type": "interrupt",
            "session_id": "session-2",
            "turn_id": "turn-2"
        }))
        .expect("deserialize interrupt op");

        assert_eq!(
            op,
            AgentOp::Interrupt {
                session_id: "session-2".to_string(),
                turn_id: Some("turn-2".to_string()),
            }
        );
    }

    #[test]
    fn agent_image_attachment_deserializes_media_type_alias() {
        let attachment: AgentImageAttachment = serde_json::from_value(serde_json::json!({
            "data": "aGVsbG8=",
            "mediaType": "image/png"
        }))
        .expect("deserialize image attachment");

        assert_eq!(attachment.media_type, "image/png");
    }

    #[test]
    fn agent_event_runtime_status_serializes_with_protocol_tag() {
        let value = serde_json::to_value(AgentEvent::RuntimeStatus {
            status: AgentRuntimeStatus {
                phase: "routing".to_string(),
                title: "等待执行窗口".to_string(),
                detail: "系统正在安排执行窗口".to_string(),
                checkpoints: vec!["并发预算 1/2".to_string()],
                metadata: None,
            },
        })
        .expect("serialize runtime status");

        assert_eq!(value["type"], "runtime_status");
        assert_eq!(value["status"]["phase"], "routing");
    }

    #[test]
    fn agent_event_provider_stream_event_serializes_current_payload() {
        let event = model_provider::provider_stream::RuntimeReplyProviderStreamEvent::SafetyBuffering(
            model_provider::safety::ProviderSafetyBufferingRuntimeEventPayload {
                kind: model_provider::safety::SAFETY_BUFFERING_RUNTIME_EVENT_KIND,
                provider: Some("openai".to_string()),
                model: Some("gpt-5-codex".to_string()),
                use_cases: vec!["policy".to_string()],
                reasons: vec!["buffering".to_string()],
                show_buffering_ui: true,
                retry_model: Some("gpt-5-mini".to_string()),
                fallback_header_model: None,
                source: model_provider::safety::ProviderSafetyBufferingRetryModelSource::PayloadRetryModel,
            },
        );
        let value = serde_json::to_value(AgentEvent::ProviderStreamEvent {
            runtime_event_kind: event.runtime_event_kind().to_string(),
            payload: event.payload_json_value(),
        })
        .expect("serialize provider stream event");

        assert_eq!(value["type"], "provider_stream_event");
        assert_eq!(value["runtime_event_kind"], "provider_safety_buffering");
        assert_eq!(value["payload"]["retryModel"], "gpt-5-mini");
        assert_eq!(value["payload"]["source"], "payload_retry_model");
        assert!(value["payload"].get("retry_model").is_none());
        assert!(value["payload"].get("fasterModel").is_none());
    }

    #[test]
    fn agent_event_serializes_provider_neutral_model_facts() {
        let server_model = serde_json::to_value(AgentEvent::ServerModel {
            model: "gpt-5-codex".to_string(),
        })
        .expect("serialize server model");
        assert_eq!(server_model["type"], "server_model");
        assert_eq!(server_model["model"], "gpt-5-codex");

        let reroute = serde_json::to_value(AgentEvent::ModelReroute {
            from_model: "gpt-5-codex".to_string(),
            to_model: "gpt-5.1-codex".to_string(),
            reason: ModelRerouteReason::HighRiskCyberActivity,
        })
        .expect("serialize model reroute");
        assert_eq!(reroute["type"], "model_reroute");
        assert_eq!(reroute["from_model"], "gpt-5-codex");
        assert_eq!(reroute["to_model"], "gpt-5.1-codex");
        assert_eq!(reroute["reason"], "high_risk_cyber_activity");

        let verification = serde_json::to_value(AgentEvent::ModelVerification {
            verifications: vec![ModelVerification::TrustedAccessForCyber],
        })
        .expect("serialize model verification");
        assert_eq!(verification["type"], "model_verification");
        assert_eq!(
            verification["verifications"],
            serde_json::json!(["trusted_access_for_cyber"])
        );

        let moderation = serde_json::to_value(AgentEvent::TurnModerationMetadata {
            metadata: serde_json::json!({ "presentation": "inline" }),
        })
        .expect("serialize turn moderation metadata");
        assert_eq!(moderation["type"], "turn_moderation_metadata");
        assert_eq!(
            moderation["metadata"],
            serde_json::json!({ "presentation": "inline" })
        );
    }

    #[test]
    fn diagnostics_runtime_status_metadata_marks_runtime_status_as_transient() {
        let metadata = build_diagnostics_runtime_status_metadata();

        assert_eq!(
            metadata.get("sourceType").and_then(Value::as_str),
            Some("runtime_status")
        );
        assert_eq!(
            metadata.get("visibility").and_then(Value::as_str),
            Some("diagnostics")
        );
        assert_eq!(
            metadata.get("persistence").and_then(Value::as_str),
            Some("transient")
        );
        assert_eq!(
            metadata
                .get("agentui")
                .and_then(|value| value.get("eventClass"))
                .and_then(Value::as_str),
            Some("run.status")
        );
    }

    #[test]
    fn agent_event_text_delta_batch_serializes_with_protocol_tag() {
        let value = serde_json::to_value(AgentEvent::TextDeltaBatch {
            text: "第一段\n".to_string(),
            chunks: vec!["第一段".to_string(), "\n".to_string()],
            boundary: TextDeltaBatchBoundary::Newline,
        })
        .expect("serialize text delta batch");

        assert_eq!(value["type"], "text_delta_batch");
        assert_eq!(value["text"], "第一段\n");
        assert_eq!(value["chunks"][0], "第一段");
        assert_eq!(value["boundary"], "newline");
    }

    #[test]
    fn agent_output_lifecycle_serializes_required_item_identity() {
        let events = [
            AgentEvent::TextStart {
                item_id: "message-1".to_string(),
            },
            AgentEvent::TextDelta {
                item_id: "message-1".to_string(),
                text: "answer".to_string(),
            },
            AgentEvent::TextEnd {
                item_id: "message-1".to_string(),
                phase: AgentMessagePhase::FinalAnswer,
            },
            AgentEvent::ReasoningSummaryDelta {
                item_id: "reasoning-1".to_string(),
                text: "inspect".to_string(),
                summary_index: 0,
            },
            AgentEvent::ReasoningSummaryPartAdded {
                item_id: "reasoning-1".to_string(),
                summary_index: 1,
            },
            AgentEvent::ReasoningContentDelta {
                item_id: "reasoning-1".to_string(),
                text: "raw".to_string(),
                content_index: 0,
            },
        ];
        let values = events
            .into_iter()
            .map(|event| serde_json::to_value(event).expect("serialize output lifecycle"))
            .collect::<Vec<_>>();

        assert_eq!(values[0]["type"], "text_start");
        assert_eq!(values[0]["itemId"], "message-1");
        assert_eq!(values[1]["type"], "text_delta");
        assert_eq!(values[1]["itemId"], "message-1");
        assert_eq!(values[2]["phase"], "final_answer");
        assert_eq!(values[2]["type"], "text_end");
        assert_eq!(values[2]["itemId"], "message-1");
        assert_eq!(values[3]["type"], "reasoning_summary_delta");
        assert_eq!(values[3]["itemId"], "reasoning-1");
        assert_eq!(values[3]["summaryIndex"], 0);
        assert_eq!(values[4]["type"], "reasoning_summary_part_added");
        assert_eq!(values[4]["itemId"], "reasoning-1");
        assert_eq!(values[4]["summaryIndex"], 1);
        assert_eq!(values[5]["type"], "reasoning_content_delta");
        assert_eq!(values[5]["itemId"], "reasoning-1");
        assert_eq!(values[5]["contentIndex"], 0);
    }
}
