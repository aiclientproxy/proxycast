//! Current provider contract for durable multi-agent control.
//!
//! Tool definitions and argument validation live here. The concrete owner is supplied per turn
//! by App Server; this crate never owns agent graph, identity, mailbox, or RuntimeCore state.

use crate::tool_definition::RuntimeToolDefinition;
use crate::tool_executor::{
    RuntimeToolExecutionError, RuntimeToolExecutionRequest, RuntimeToolExecutionResult,
    RuntimeToolPolicyErrorKind,
};
use agent_protocol::{CollabAgentState, ThreadId};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

pub const SPAWN_AGENT_TOOL_NAME: &str = "spawn_agent";
pub const SEND_MESSAGE_TOOL_NAME: &str = "send_message";
pub const FOLLOWUP_TASK_TOOL_NAME: &str = "followup_task";
pub const WAIT_AGENT_TOOL_NAME: &str = "wait_agent";
pub const INTERRUPT_AGENT_TOOL_NAME: &str = "interrupt_agent";
pub const LIST_AGENTS_TOOL_NAME: &str = "list_agents";
const DEFAULT_WAIT_TIMEOUT_MS: u64 = 30_000;
const MIN_WAIT_TIMEOUT_MS: u64 = 10_000;
const MAX_WAIT_TIMEOUT_MS: u64 = 3_600_000;
const MAX_SPAWN_AGENT_MODEL_OVERRIDES: usize = 5;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpawnAgentModelOption {
    pub model: String,
    pub description: String,
    pub supported_reasoning_efforts: Vec<String>,
    pub default_reasoning_effort: Option<String>,
    pub service_tiers: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpawnAgentToolOptions {
    pub available_models: Vec<SpawnAgentModelOption>,
}

pub fn is_agent_control_tool_name(name: &str) -> bool {
    matches!(
        name.trim(),
        SPAWN_AGENT_TOOL_NAME
            | SEND_MESSAGE_TOOL_NAME
            | FOLLOWUP_TASK_TOOL_NAME
            | WAIT_AGENT_TOOL_NAME
            | INTERRUPT_AGENT_TOOL_NAME
            | LIST_AGENTS_TOOL_NAME
    )
}

pub fn agent_control_tool_definitions() -> Vec<RuntimeToolDefinition> {
    agent_control_tool_definitions_with_options(&SpawnAgentToolOptions::default())
}

pub fn agent_control_tool_definitions_with_options(
    options: &SpawnAgentToolOptions,
) -> Vec<RuntimeToolDefinition> {
    let spawn_description = format!(
        "Spawn a child agent in the current durable agent tree.\n\n{}",
        spawn_agent_models_description(&options.available_models)
    );
    vec![
        RuntimeToolDefinition::new(
            SPAWN_AGENT_TOOL_NAME,
            spawn_description,
            json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "task_name": {
                        "type": "string",
                        "description": "Task name for the new agent. Use lowercase letters, digits, and underscores."
                    },
                    "message": { "type": "string" },
                    "fork_turns": {
                        "type": "string",
                        "description": "Optional number of turns to fork. Defaults to `all`. Use `none`, `all`, or a positive integer string such as `3`."
                    },
                    "model": {
                        "type": "string",
                        "description": "Model override for the new agent. Omit to inherit the parent model."
                    },
                    "reasoning_effort": {
                        "type": "string",
                        "description": "Reasoning effort override for the new agent. Omit to inherit the parent effort."
                    },
                    "service_tier": {
                        "type": "string",
                        "description": "Service tier override for the new agent. Omit to inherit the parent tier."
                    }
                },
                "required": ["task_name", "message"]
            }),
        ),
        RuntimeToolDefinition::new(
            SEND_MESSAGE_TOOL_NAME,
            "Queue a message for an agent in the current durable agent tree.",
            message_input_schema(),
        ),
        RuntimeToolDefinition::new(
            FOLLOWUP_TASK_TOOL_NAME,
            "Send a follow-up task and trigger the target child agent.",
            message_input_schema(),
        ),
        RuntimeToolDefinition::new(
            WAIT_AGENT_TOOL_NAME,
            "Wait for durable mailbox activity from the current agent tree.",
            json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "timeout_ms": {
                        "type": "integer",
                        "minimum": MIN_WAIT_TIMEOUT_MS,
                        "maximum": MAX_WAIT_TIMEOUT_MS
                    }
                }
            }),
        ),
        RuntimeToolDefinition::new(
            INTERRUPT_AGENT_TOOL_NAME,
            "Interrupt a child agent's active turn while keeping its durable graph edge open.",
            target_input_schema(),
        ),
        RuntimeToolDefinition::new(
            LIST_AGENTS_TOOL_NAME,
            "List agents in the current durable agent tree.",
            json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "path_prefix": { "type": "string" }
                }
            }),
        ),
    ]
}

fn spawn_agent_models_description(models: &[SpawnAgentModelOption]) -> String {
    if models.is_empty() {
        return "No picker-visible model overrides are currently loaded.".to_string();
    }
    let model_descriptions = models
        .iter()
        .take(MAX_SPAWN_AGENT_MODEL_OVERRIDES)
        .map(|model| {
            let reasoning_efforts = model
                .supported_reasoning_efforts
                .iter()
                .map(|effort| {
                    if model.default_reasoning_effort.as_deref() == Some(effort.as_str()) {
                        format!("{effort} (default)")
                    } else {
                        effort.clone()
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            let reasoning_suffix = if reasoning_efforts.is_empty() {
                String::new()
            } else {
                format!(" Reasoning efforts: {reasoning_efforts}.")
            };
            let service_tiers = model.service_tiers.join(", ");
            let service_tier_suffix = if service_tiers.is_empty() {
                String::new()
            } else {
                format!(" Service tiers: {service_tiers}.")
            };
            format!(
                "- `{}`: {}{reasoning_suffix}{service_tier_suffix}",
                model.model, model.description
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "Available model overrides (optional; inherited parent model is preferred):\n{model_descriptions}"
    )
}

fn message_input_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "target": { "type": "string" },
            "message": { "type": "string" }
        },
        "required": ["target", "message"]
    })
}

fn target_input_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "target": { "type": "string" }
        },
        "required": ["target"]
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentControlCaller {
    pub session_id: String,
    pub thread_id: String,
    pub turn_id: String,
    pub call_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpawnAgentForkMode {
    None,
    FullHistory,
    LastNTurns(usize),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpawnAgentModelOverrides {
    pub model: Option<String>,
    pub reasoning_effort: Option<String>,
    pub service_tier: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentControlCommand {
    SpawnAgent {
        task_name: String,
        message: String,
        fork_mode: SpawnAgentForkMode,
        model_overrides: SpawnAgentModelOverrides,
    },
    SendMessage {
        target: String,
        message: String,
    },
    FollowupTask {
        target: String,
        message: String,
    },
    WaitAgent {
        timeout_ms: u64,
    },
    InterruptAgent {
        target: String,
    },
    ListAgents {
        path_prefix: Option<String>,
    },
}

#[derive(Clone, Debug)]
pub struct AgentControlGatewayRequest {
    pub caller: AgentControlCaller,
    pub command: AgentControlCommand,
    pub cancel_token: Option<CancellationToken>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AgentControlGatewayResult {
    pub output: Value,
    pub projection_facts: Vec<SubAgentProjectionFact>,
    pub state_facts: Vec<AgentStateProjectionFact>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubAgentProjectionActivity {
    Started,
    Interacted,
    Interrupted,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubAgentProjectionFact {
    pub target_thread_id: ThreadId,
    pub activity: SubAgentProjectionActivity,
    pub detail: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentStateProjectionFact {
    pub target_thread_id: ThreadId,
    pub state: CollabAgentState,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentControlGatewayError {
    message: String,
}

impl AgentControlGatewayError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for AgentControlGatewayError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for AgentControlGatewayError {}

#[async_trait]
pub trait AgentControlGateway: Send + Sync {
    async fn execute(
        &self,
        request: AgentControlGatewayRequest,
    ) -> Result<AgentControlGatewayResult, AgentControlGatewayError>;
}

#[derive(Clone)]
pub struct AgentControlGatewayHandle {
    gateway: Arc<dyn AgentControlGateway>,
    tool_options: SpawnAgentToolOptions,
}

impl AgentControlGatewayHandle {
    pub fn new(gateway: Arc<dyn AgentControlGateway>) -> Self {
        Self {
            gateway,
            tool_options: SpawnAgentToolOptions::default(),
        }
    }

    pub fn with_tool_options(mut self, tool_options: SpawnAgentToolOptions) -> Self {
        self.tool_options = tool_options;
        self
    }

    pub fn gateway(&self) -> &dyn AgentControlGateway {
        self.gateway.as_ref()
    }

    pub fn tool_definitions(&self) -> Vec<RuntimeToolDefinition> {
        agent_control_tool_definitions_with_options(&self.tool_options)
    }
}

impl std::fmt::Debug for AgentControlGatewayHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("AgentControlGatewayHandle(..)")
    }
}

pub async fn execute_agent_control_tool(
    gateway: &dyn AgentControlGateway,
    thread_id: &str,
    request: RuntimeToolExecutionRequest<'_>,
) -> Option<Result<RuntimeToolExecutionResult, RuntimeToolExecutionError>> {
    let command = match parse_command(request.tool_name, request.params) {
        Some(Ok(command)) => command,
        Some(Err(error)) => return Some(Err(error)),
        None => return None,
    };
    let Some(identity) = request.context.tool_identity() else {
        return Some(Err(agent_control_execution_error(
            "agent control requires canonical tool identity",
            "agent_control_identity_missing",
        )));
    };
    let turn_id = identity.turn_id().trim();
    let call_id = identity.call_id().trim();
    let thread_id = thread_id.trim();
    if turn_id.is_empty() || call_id.is_empty() || thread_id.is_empty() {
        return Some(Err(agent_control_execution_error(
            "agent control requires canonical caller identity",
            "agent_control_identity_invalid",
        )));
    }
    let result = gateway
        .execute(AgentControlGatewayRequest {
            caller: AgentControlCaller {
                session_id: request.context.session_id().to_string(),
                thread_id: thread_id.to_string(),
                turn_id: turn_id.to_string(),
                call_id: call_id.to_string(),
            },
            command,
            cancel_token: request.context.cancel_token().cloned(),
        })
        .await
        .map_err(|error| {
            agent_control_execution_error(
                format!("agent control failed: {error}"),
                "agent_control_gateway_failed",
            )
        });
    Some(result.and_then(runtime_execution_result))
}

fn runtime_execution_result(
    result: AgentControlGatewayResult,
) -> Result<RuntimeToolExecutionResult, RuntimeToolExecutionError> {
    let AgentControlGatewayResult {
        output,
        projection_facts,
        state_facts,
    } = result;
    let output =
        serde_json::to_string(&output).map_err(agent_control_result_serialization_error)?;
    Ok(
        RuntimeToolExecutionResult::new(true, output, None, Default::default())
            .with_agent_control_projection_facts(projection_facts)
            .with_agent_control_state_facts(state_facts),
    )
}

fn agent_control_result_serialization_error(error: serde_json::Error) -> RuntimeToolExecutionError {
    agent_control_execution_error(
        format!("agent control result serialization failed: {error}"),
        "agent_control_result_invalid",
    )
}

fn parse_command(
    tool_name: &str,
    params: &Value,
) -> Option<Result<AgentControlCommand, RuntimeToolExecutionError>> {
    match tool_name.trim() {
        SPAWN_AGENT_TOOL_NAME => Some(parse_spawn(params)),
        SEND_MESSAGE_TOOL_NAME => {
            Some(
                parse_message(params).map(|input| AgentControlCommand::SendMessage {
                    target: input.target,
                    message: input.message,
                }),
            )
        }
        FOLLOWUP_TASK_TOOL_NAME => {
            Some(
                parse_message(params).map(|input| AgentControlCommand::FollowupTask {
                    target: input.target,
                    message: input.message,
                }),
            )
        }
        WAIT_AGENT_TOOL_NAME => Some(parse_wait(params)),
        INTERRUPT_AGENT_TOOL_NAME => {
            Some(
                parse_target(params).map(|input| AgentControlCommand::InterruptAgent {
                    target: input.target,
                }),
            )
        }
        LIST_AGENTS_TOOL_NAME => Some(parse_list(params)),
        _ => None,
    }
}

fn parse_spawn(params: &Value) -> Result<AgentControlCommand, RuntimeToolExecutionError> {
    let input: SpawnAgentInput = parse_input(params)?;
    let task_name = required_nonempty(input.task_name, "task_name")?;
    if matches!(task_name.as_str(), "root" | "." | "..")
        || !task_name
            .chars()
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
    {
        return Err(agent_control_execution_error(
            "task_name must use only lowercase letters, digits, and underscores",
            "agent_control_invalid_params",
        ));
    }
    Ok(AgentControlCommand::SpawnAgent {
        task_name,
        message: required_nonempty(input.message, "message")?,
        fork_mode: parse_spawn_fork_mode(input.fork_turns)?,
        model_overrides: SpawnAgentModelOverrides {
            model: optional_nonempty(input.model, "model")?,
            reasoning_effort: optional_nonempty(input.reasoning_effort, "reasoning_effort")?,
            service_tier: optional_nonempty(input.service_tier, "service_tier")?,
        },
    })
}

fn parse_spawn_fork_mode(
    fork_turns: Option<String>,
) -> Result<SpawnAgentForkMode, RuntimeToolExecutionError> {
    let fork_turns = fork_turns
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("all");
    if fork_turns.eq_ignore_ascii_case("none") {
        return Ok(SpawnAgentForkMode::None);
    }
    if fork_turns.eq_ignore_ascii_case("all") {
        return Ok(SpawnAgentForkMode::FullHistory);
    }
    let last_n_turns = fork_turns
        .parse::<usize>()
        .map_err(|_| invalid_spawn_fork_turns())?;
    if last_n_turns == 0 {
        return Err(invalid_spawn_fork_turns());
    }
    Ok(SpawnAgentForkMode::LastNTurns(last_n_turns))
}

fn invalid_spawn_fork_turns() -> RuntimeToolExecutionError {
    agent_control_execution_error(
        "fork_turns must be `none`, `all`, or a positive integer string",
        "agent_control_invalid_params",
    )
}

fn parse_message(params: &Value) -> Result<MessageInput, RuntimeToolExecutionError> {
    let input: MessageInput = parse_input(params)?;
    Ok(MessageInput {
        target: required_nonempty(input.target, "target")?,
        message: required_nonempty(input.message, "message")?,
    })
}

fn parse_target(params: &Value) -> Result<TargetInput, RuntimeToolExecutionError> {
    let input: TargetInput = parse_input(params)?;
    Ok(TargetInput {
        target: required_nonempty(input.target, "target")?,
    })
}

fn parse_wait(params: &Value) -> Result<AgentControlCommand, RuntimeToolExecutionError> {
    let input: WaitAgentInput = parse_input(params)?;
    let timeout_ms = input.timeout_ms.unwrap_or(DEFAULT_WAIT_TIMEOUT_MS);
    if timeout_ms < MIN_WAIT_TIMEOUT_MS {
        return Err(agent_control_execution_error(
            format!("timeout_ms must be at least {MIN_WAIT_TIMEOUT_MS}"),
            "agent_control_invalid_params",
        ));
    }
    if timeout_ms > MAX_WAIT_TIMEOUT_MS {
        return Err(agent_control_execution_error(
            format!("timeout_ms must be at most {MAX_WAIT_TIMEOUT_MS}"),
            "agent_control_invalid_params",
        ));
    }
    Ok(AgentControlCommand::WaitAgent { timeout_ms })
}

fn parse_list(params: &Value) -> Result<AgentControlCommand, RuntimeToolExecutionError> {
    let input: ListAgentsInput = parse_input(params)?;
    let path_prefix = input
        .path_prefix
        .map(|value| required_nonempty(value, "path_prefix"))
        .transpose()?;
    Ok(AgentControlCommand::ListAgents { path_prefix })
}

fn parse_input<T: for<'de> Deserialize<'de>>(
    params: &Value,
) -> Result<T, RuntimeToolExecutionError> {
    serde_json::from_value(params.clone()).map_err(|error| {
        agent_control_execution_error(
            format!("agent control parameters are invalid: {error}"),
            "agent_control_invalid_params",
        )
    })
}

fn required_nonempty(value: String, field: &str) -> Result<String, RuntimeToolExecutionError> {
    let value = value.trim().to_string();
    (!value.is_empty()).then_some(value).ok_or_else(|| {
        agent_control_execution_error(
            format!("{field} is required"),
            "agent_control_invalid_params",
        )
    })
}

fn optional_nonempty(
    value: Option<String>,
    field: &str,
) -> Result<Option<String>, RuntimeToolExecutionError> {
    value
        .map(|value| required_nonempty(value, field))
        .transpose()
}

fn agent_control_execution_error(
    message: impl Into<String>,
    code: &str,
) -> RuntimeToolExecutionError {
    RuntimeToolExecutionError::new(
        message,
        Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
            code.to_string(),
        )),
    )
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpawnAgentInput {
    task_name: String,
    message: String,
    fork_turns: Option<String>,
    model: Option<String>,
    reasoning_effort: Option<String>,
    service_tier: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MessageInput {
    target: String,
    message: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetInput {
    target: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WaitAgentInput {
    timeout_ms: Option<u64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ListAgentsInput {
    path_prefix: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_executor::{
        RuntimeToolExecutionContext, RuntimeToolExecutionContextInput, RuntimeToolExecutionIdentity,
    };
    use std::path::PathBuf;

    #[derive(Default)]
    struct RecordingGateway {
        requests: std::sync::Mutex<Vec<AgentControlGatewayRequest>>,
    }

    #[async_trait]
    impl AgentControlGateway for RecordingGateway {
        async fn execute(
            &self,
            request: AgentControlGatewayRequest,
        ) -> Result<AgentControlGatewayResult, AgentControlGatewayError> {
            self.requests
                .lock()
                .expect("requests mutex poisoned")
                .push(request);
            Ok(AgentControlGatewayResult {
                output: json!({ "accepted": true }),
                projection_facts: Vec::new(),
                state_facts: Vec::new(),
            })
        }
    }

    fn request<'a>(
        tool_name: &'a str,
        params: &'a Value,
        context: &'a RuntimeToolExecutionContext,
    ) -> RuntimeToolExecutionRequest<'a> {
        RuntimeToolExecutionRequest {
            tool_name,
            params,
            context,
            turn_context: None,
        }
    }

    fn context() -> RuntimeToolExecutionContext {
        RuntimeToolExecutionContext::new(RuntimeToolExecutionContextInput {
            working_directory: PathBuf::from("/workspace"),
            session_id: "session-root".to_string(),
            cancel_token: None,
            workspace_sandbox: None,
        })
        .with_tool_identity(RuntimeToolExecutionIdentity::new("call-1", "turn-1"))
    }

    #[test]
    fn exposes_only_v2_agent_control_tools() {
        let names = agent_control_tool_definitions()
            .into_iter()
            .map(|definition| definition.name)
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            vec![
                "spawn_agent",
                "send_message",
                "followup_task",
                "wait_agent",
                "interrupt_agent",
                "list_agents",
            ]
        );
        assert!(is_agent_control_tool_name(SPAWN_AGENT_TOOL_NAME));
        assert!(!is_agent_control_tool_name("TeamCreate"));
    }

    #[test]
    fn spawn_description_uses_catalog_model_effort_and_service_tiers() {
        let definitions = agent_control_tool_definitions_with_options(&SpawnAgentToolOptions {
            available_models: vec![SpawnAgentModelOption {
                model: "gpt-5.6-sol".to_string(),
                description: "Latest frontier agentic coding model.".to_string(),
                supported_reasoning_efforts: vec!["low".to_string(), "high".to_string()],
                default_reasoning_effort: Some("low".to_string()),
                service_tiers: vec!["priority".to_string()],
            }],
        });
        let spawn = definitions
            .iter()
            .find(|definition| definition.name == SPAWN_AGENT_TOOL_NAME)
            .expect("spawn definition");

        assert!(spawn.description.contains(
            "Available model overrides (optional; inherited parent model is preferred):"
        ));
        assert!(spawn.description.contains("`gpt-5.6-sol`"));
        assert!(spawn
            .description
            .contains("Reasoning efforts: low (default), high."));
        assert!(spawn.description.contains("Service tiers: priority."));
    }

    #[tokio::test]
    async fn dispatches_typed_spawn_without_legacy_aliases() {
        let gateway = RecordingGateway::default();
        let params = json!({
            "task_name": "research",
            "message": "inspect the plan",
            "model": "gpt-5.6-sol",
            "reasoning_effort": "high",
            "service_tier": "priority",
        });
        let context = context();

        let result = execute_agent_control_tool(
            &gateway,
            "thread-root",
            request(SPAWN_AGENT_TOOL_NAME, &params, &context),
        )
        .await
        .expect("current agent control tool")
        .expect("gateway result");

        assert_eq!(result.output, "{\"accepted\":true}");
        let requests = gateway.requests.lock().expect("requests mutex poisoned");
        assert_eq!(requests.len(), 1);
        assert_eq!(
            requests[0].caller,
            AgentControlCaller {
                session_id: "session-root".to_string(),
                thread_id: "thread-root".to_string(),
                turn_id: "turn-1".to_string(),
                call_id: "call-1".to_string(),
            }
        );
        assert_eq!(
            requests[0].command,
            AgentControlCommand::SpawnAgent {
                task_name: "research".to_string(),
                message: "inspect the plan".to_string(),
                fork_mode: SpawnAgentForkMode::FullHistory,
                model_overrides: SpawnAgentModelOverrides {
                    model: Some("gpt-5.6-sol".to_string()),
                    reasoning_effort: Some("high".to_string()),
                    service_tier: Some("priority".to_string()),
                },
            }
        );
        assert!(requests[0].cancel_token.is_none());
        assert!(execute_agent_control_tool(
            &gateway,
            "thread-root",
            request("TeamCreate", &params, &context),
        )
        .await
        .is_none());
    }

    #[tokio::test]
    async fn validates_codex_v2_spawn_fork_turns() {
        let gateway = RecordingGateway::default();
        let context = context();
        for (value, expected) in [
            ("none", SpawnAgentForkMode::None),
            ("all", SpawnAgentForkMode::FullHistory),
            ("3", SpawnAgentForkMode::LastNTurns(3)),
            (" 2 ", SpawnAgentForkMode::LastNTurns(2)),
            ("", SpawnAgentForkMode::FullHistory),
        ] {
            let params = json!({
                "task_name": "research",
                "message": "inspect the plan",
                "fork_turns": value,
            });
            execute_agent_control_tool(
                &gateway,
                "thread-root",
                request(SPAWN_AGENT_TOOL_NAME, &params, &context),
            )
            .await
            .expect("current tool")
            .expect("valid fork_turns");
            let actual = gateway
                .requests
                .lock()
                .expect("requests mutex poisoned")
                .last()
                .expect("gateway request")
                .command
                .clone();
            assert_eq!(
                actual,
                AgentControlCommand::SpawnAgent {
                    task_name: "research".to_string(),
                    message: "inspect the plan".to_string(),
                    fork_mode: expected,
                    model_overrides: SpawnAgentModelOverrides::default(),
                }
            );
        }

        for value in ["0", "banana", "-1"] {
            let params = json!({
                "task_name": "research",
                "message": "inspect the plan",
                "fork_turns": value,
            });
            let error = execute_agent_control_tool(
                &gateway,
                "thread-root",
                request(SPAWN_AGENT_TOOL_NAME, &params, &context),
            )
            .await
            .expect("current tool")
            .expect_err("invalid fork_turns");
            assert!(error
                .message()
                .contains("fork_turns must be `none`, `all`, or a positive integer string"));
        }

        let legacy = json!({
            "task_name": "research",
            "message": "inspect the plan",
            "fork_context": true,
        });
        assert!(execute_agent_control_tool(
            &gateway,
            "thread-root",
            request(SPAWN_AGENT_TOOL_NAME, &legacy, &context),
        )
        .await
        .expect("current tool")
        .is_err());
    }

    #[tokio::test]
    async fn rejects_ambiguous_or_invalid_parameters() {
        let gateway = RecordingGateway::default();
        let context = context();
        for task_name in ["nested/agent", "Worker", "worker-name", "root", ".", ".."] {
            let invalid = json!({ "task_name": task_name, "message": "work" });
            let error = execute_agent_control_tool(
                &gateway,
                "thread-root",
                request(SPAWN_AGENT_TOOL_NAME, &invalid, &context),
            )
            .await
            .expect("current tool")
            .expect_err("invalid task name");
            assert_eq!(
                error.policy_kind(),
                Some(&RuntimeToolPolicyErrorKind::ExecutionFailed(
                    "agent_control_invalid_params".to_string()
                ))
            );
        }

        let unknown = json!({ "target": "child", "message": "continue", "legacy": true });
        assert!(execute_agent_control_tool(
            &gateway,
            "thread-root",
            request(SEND_MESSAGE_TOOL_NAME, &unknown, &context),
        )
        .await
        .expect("current tool")
        .is_err());
    }

    #[test]
    fn wait_agent_matches_codex_v2_timeout_contract() {
        let definition = agent_control_tool_definitions()
            .into_iter()
            .find(|definition| definition.name == WAIT_AGENT_TOOL_NAME)
            .expect("wait_agent definition");
        assert_eq!(
            definition.input_schema["properties"]["timeout_ms"]["minimum"],
            json!(MIN_WAIT_TIMEOUT_MS)
        );
        assert_eq!(
            definition.input_schema["properties"]["timeout_ms"]["maximum"],
            json!(MAX_WAIT_TIMEOUT_MS)
        );

        assert_eq!(
            parse_wait(&json!({})).expect("default timeout"),
            AgentControlCommand::WaitAgent {
                timeout_ms: DEFAULT_WAIT_TIMEOUT_MS,
            }
        );
        assert_eq!(
            parse_wait(&json!({ "timeout_ms": MIN_WAIT_TIMEOUT_MS })).expect("minimum timeout"),
            AgentControlCommand::WaitAgent {
                timeout_ms: MIN_WAIT_TIMEOUT_MS,
            }
        );
        assert_eq!(
            parse_wait(&json!({ "timeout_ms": MAX_WAIT_TIMEOUT_MS })).expect("maximum timeout"),
            AgentControlCommand::WaitAgent {
                timeout_ms: MAX_WAIT_TIMEOUT_MS,
            }
        );

        for timeout_ms in [MIN_WAIT_TIMEOUT_MS - 1, MAX_WAIT_TIMEOUT_MS + 1] {
            let error =
                parse_wait(&json!({ "timeout_ms": timeout_ms })).expect_err("out-of-range timeout");
            assert_eq!(
                error.policy_kind(),
                Some(&RuntimeToolPolicyErrorKind::ExecutionFailed(
                    "agent_control_invalid_params".to_string()
                ))
            );
        }
    }

    #[test]
    fn transports_typed_projection_facts_outside_model_visible_output() {
        let expected_activity = vec![
            SubAgentProjectionFact {
                target_thread_id: ThreadId::new("thread-child"),
                activity: SubAgentProjectionActivity::Started,
                detail: Some("/root/research".to_string()),
            },
            SubAgentProjectionFact {
                target_thread_id: ThreadId::new("thread-child"),
                activity: SubAgentProjectionActivity::Interacted,
                detail: None,
            },
            SubAgentProjectionFact {
                target_thread_id: ThreadId::new("thread-child"),
                activity: SubAgentProjectionActivity::Interrupted,
                detail: None,
            },
        ];
        let expected_states = vec![AgentStateProjectionFact {
            target_thread_id: ThreadId::new("thread-child"),
            state: CollabAgentState {
                status: agent_protocol::CollabAgentStatus::Completed,
                message: None,
            },
        }];
        let result = runtime_execution_result(AgentControlGatewayResult {
            output: json!({ "accepted": true }),
            projection_facts: expected_activity.clone(),
            state_facts: expected_states.clone(),
        })
        .expect("runtime projection");

        assert_eq!(result.output, "{\"accepted\":true}");
        assert_eq!(result.agent_control_projection_facts, expected_activity);
        assert_eq!(result.agent_control_state_facts, expected_states);
        assert!(result.metadata.is_empty());

        let normalized =
            crate::tool_result_projection::NormalizedToolOutput::from_execution_outcome(
                crate::tool_executor::RuntimeToolExecutionOutcome::Result(result),
                1,
            );
        assert_eq!(normalized.agent_control_projection_facts, expected_activity);
        assert_eq!(normalized.agent_control_state_facts, expected_states);
        let serialized = serde_json::to_value(normalized).expect("serialize normalized output");
        assert!(serialized.get("agent_control_projection_facts").is_none());
        assert!(serialized.get("agent_control_state_facts").is_none());
    }

    #[test]
    fn omits_projection_metadata_when_gateway_returns_no_facts() {
        let result = runtime_execution_result(AgentControlGatewayResult {
            output: json!({ "agents": [] }),
            projection_facts: Vec::new(),
            state_facts: Vec::new(),
        })
        .expect("runtime projection");

        assert_eq!(result.output, "{\"agents\":[]}");
        assert!(result.agent_control_projection_facts.is_empty());
        assert!(result.metadata.is_empty());
    }
}
