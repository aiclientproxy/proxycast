use super::RuntimeHostContext;
use agent_runtime::reply_input::RuntimeReplyInput;

#[derive(Debug, Clone)]
pub struct ExecutionRequest {
    pub host: RuntimeHostContext,
    pub session: app_server_protocol::AgentSession,
    pub turn: app_server_protocol::AgentTurn,
    /// Canonical Thread lineage captured at turn admission.
    pub forked_from_thread_id: Option<String>,
    pub input: RuntimeReplyInput,
    pub runtime_options: Option<app_server_protocol::RuntimeOptions>,
    pub expected_output: Option<serde_json::Value>,
    pub structured_output: Option<app_server_protocol::StructuredOutputContract>,
    pub output_schema: Option<serde_json::Value>,
    pub event_name: Option<String>,
    pub queued_turn_id: Option<String>,
    pub queue_if_busy: bool,
    pub skip_pre_submit_resume: bool,
    /// Per-turn capability supplied by RuntimeCore for the current provider execution only.
    pub agent_control_gateway: Option<tool_runtime::agent_control::AgentControlGatewayHandle>,
    /// RuntimeCore-owned sampling boundary for later reminders in the same turn.
    pub rollout_budget_reminder_source:
        Option<agent_runtime::session_config::RolloutBudgetReminderSourceHandle>,
}

impl ExecutionRequest {
    /// RuntimeRequest 是 Turn 执行配置和运行时元数据的唯一 owner。
    pub fn runtime_request(&self) -> Option<&app_server_protocol::RuntimeRequest> {
        self.runtime_options
            .as_ref()
            .and_then(app_server_protocol::RuntimeOptions::runtime_request)
    }

    pub fn runtime_metadata(&self) -> Option<&serde_json::Value> {
        self.runtime_options
            .as_ref()
            .and_then(app_server_protocol::RuntimeOptions::runtime_metadata)
    }

    pub fn provider_preference(&self) -> Option<&str> {
        self.runtime_options
            .as_ref()
            .and_then(app_server_protocol::RuntimeOptions::provider_preference)
    }

    pub fn model_preference(&self) -> Option<&str> {
        self.runtime_options
            .as_ref()
            .and_then(app_server_protocol::RuntimeOptions::model_preference)
    }
}
