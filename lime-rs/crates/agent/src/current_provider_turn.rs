//! 配置 provider 的 current 回合适配器。
//!
//! provider 网络协议由 `model-provider` lower，采样和 tool-result transcript 由
//! `agent-runtime::provider_turn` 维护；本模块只连接 Lime 的动态工具注册表、MCP
//! registry 和 App Server 已消费的事件协议。这里不依赖 Agent。

use crate::credential_bridge::ConfiguredReplyProvider;
use crate::model_request_policy::{
    input_modality_policy_allows_image_input, input_modality_policy_from_turn_context,
    runtime_reply_model_request_policy_from_turn_context,
};
use crate::protocol::{AgentEvent, AgentTokenUsage};
use crate::request_tool_policy::{
    is_same_tool, merge_system_prompt_with_request_tool_policy, ReplyAttemptError,
    RequestToolPolicy, StreamReplyExecution, WebSearchExecutionTracker,
};
use crate::runtime_state::AgentRuntimeState;
use crate::write_artifact_events::WriteArtifactEventEmitter;
use agent_protocol::{ItemStatus, ThreadId, ThreadItemPayload};
use agent_runtime::provider_turn::{
    run_current_provider_turn, CurrentProviderTurnEvent, CurrentProviderTurnInput,
};
use agent_runtime::reply_input::RuntimeReplyInput;
use agent_runtime::session_config::AgentSessionConfig;
use agent_runtime::session_loop::RuntimeSessionInputHandle;
#[cfg(test)]
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
#[cfg(test)]
use tool_runtime::tool_lifecycle::{ToolLifecycleEmitter, ToolLifecycleEvent, ToolLifecyclePhase};

#[cfg(test)]
mod agent_control_tests;
mod dynamic_tool_bridge;
#[cfg(test)]
mod input_tests;
mod mcp_step_snapshot;
mod structured_input;
mod tool_executor;
mod tool_lifecycle_emitter;

use structured_input::{
    prepare_image_inputs_for_model, skill_snapshot_from_turn_context, structured_input_context,
    user_message,
};
use tool_lifecycle_emitter::CurrentTurnToolLifecycleEmitter;

#[cfg(test)]
use tool_executor::{action_scope, mcp_call_scope, project_call_result};

pub(crate) async fn stream_current_provider_turn<F>(
    state: &AgentRuntimeState,
    provider: ConfiguredReplyProvider,
    input: RuntimeReplyInput,
    mut initial_messages: Vec<model_provider::current_client::CurrentProviderMessage>,
    working_directory: Option<&Path>,
    mut session_config: AgentSessionConfig,
    cancel_token: Option<CancellationToken>,
    pending_input: Option<RuntimeSessionInputHandle>,
    policy: &RequestToolPolicy,
    agent_control_gateway: Option<tool_runtime::agent_control::AgentControlGatewayHandle>,
    mut on_event: F,
) -> Result<StreamReplyExecution, ReplyAttemptError>
where
    F: FnMut(&AgentEvent) + Send,
{
    let supports_image_input = input_modality_policy_allows_image_input(
        input_modality_policy_from_turn_context(session_config.turn_context.as_ref()).as_ref(),
    );
    prepare_image_inputs_for_model(&input, &mut initial_messages, supports_image_input)?;
    session_config.system_prompt =
        merge_system_prompt_with_request_tool_policy(session_config.system_prompt.take(), policy);
    let session_id = session_config.id.clone();
    let thread_id = session_config
        .thread_id
        .clone()
        .filter(|thread_id| !thread_id.trim().is_empty())
        .ok_or_else(|| {
            ReplyAttemptError::new(
                "Current provider turn requires a canonical thread_id",
                false,
            )
        })?;
    let model_request_policy =
        runtime_reply_model_request_policy_from_turn_context(session_config.turn_context.as_ref());
    let skill_snapshot = skill_snapshot_from_turn_context(session_config.turn_context.as_ref());
    if let Some(message) = user_message(&input) {
        initial_messages.push(message);
    }
    let structured_context = structured_input_context(&input, skill_snapshot.as_ref());
    let emitted_pre_provider_warning = !structured_context.warnings.is_empty();
    initial_messages.extend(structured_context.messages);
    for warning in structured_context.warnings {
        on_event(&AgentEvent::Warning {
            code: Some(warning.code.to_string()),
            message: warning.message,
        });
    }
    let provider_name = provider.runtime_handle().provider_name().to_string();
    let provider_trace_metadata = provider.runtime_handle().provider_trace_metadata();
    let mut artifact_events = WriteArtifactEventEmitter::new(session_config.id.clone());
    let mut usage = None;
    let mut web_search_tracker = WebSearchExecutionTracker::default();
    let (host_event_sender, mut host_event_receiver) = mpsc::unbounded_channel();
    let (agent_event_sender, mut agent_event_receiver) = mpsc::unbounded_channel();
    let mcp_tool_routes = mcp_step_snapshot::McpToolRoutes::default();
    let dynamic_tool_routes = mcp_step_snapshot::DynamicToolRoutes::default();
    let tool_step_snapshot_source = mcp_step_snapshot::current_tool_step_snapshot_source(
        state.clone(),
        policy.clone(),
        session_config.turn_context.clone(),
        agent_event_sender,
        session_id.clone(),
        ThreadId::new(thread_id.clone()),
        agent_control_gateway,
        pending_input.clone(),
        mcp_tool_routes.clone(),
        dynamic_tool_routes.clone(),
    );
    let lifecycle_emitter = Arc::new(CurrentTurnToolLifecycleEmitter::with_tool_routes(
        host_event_sender.clone(),
        session_id,
        thread_id,
        mcp_tool_routes,
        dynamic_tool_routes,
    ));

    let turn_future = run_current_provider_turn(
        CurrentProviderTurnInput {
            provider: provider.client(),
            provider_trace_metadata: Some(provider_trace_metadata),
            session_config,
            initial_messages,
            tool_step_snapshot_source,
            hook_snapshot_source: None,
            model_request_policy,
            tool_lifecycle_emitter: lifecycle_emitter,
            working_directory: working_directory
                .map(Path::to_path_buf)
                .unwrap_or_else(default_working_directory),
            cancel_token,
            pending_input,
        },
        move |event| {
            let _ = host_event_sender.send(CurrentTurnHostEvent::Provider(event));
        },
    );
    tokio::pin!(turn_future);
    let execution = loop {
        tokio::select! {
            biased;
            Some(event) = host_event_receiver.recv() => {
                handle_ordered_host_event(event, &mut agent_event_receiver, &provider_name, policy, &mut artifact_events, &mut web_search_tracker, &mut usage, &mut on_event);
            }
            result = &mut turn_future => {
                drain_ready_turn_events(&mut host_event_receiver, &mut agent_event_receiver, &provider_name, policy, &mut artifact_events, &mut web_search_tracker, &mut usage, &mut on_event);
                break result;
            }
            Some(event) = agent_event_receiver.recv() => on_event(&event),
        }
    };
    let execution = execution.map_err(|error| {
        if emitted_pre_provider_warning {
            error.suppress_reroute()
        } else {
            error
        }
    })?;

    if !execution.cancelled {
        web_search_tracker
            .validate_web_search_requirement(policy)
            .map_err(|message| ReplyAttemptError::new(message, execution.emitted_any))?;
        on_event(&AgentEvent::Done {
            usage: usage.map(project_usage),
        });
    }
    Ok(execution)
}

enum CurrentTurnHostEvent {
    Provider(CurrentProviderTurnEvent),
    ToolLifecycle(AgentEvent),
}

fn handle_ordered_host_event<F>(
    event: CurrentTurnHostEvent,
    agent_event_receiver: &mut mpsc::UnboundedReceiver<AgentEvent>,
    provider_name: &str,
    policy: &RequestToolPolicy,
    artifact_events: &mut WriteArtifactEventEmitter,
    web_search_tracker: &mut WebSearchExecutionTracker,
    usage: &mut Option<model_provider::current_client::CurrentProviderUsage>,
    on_event: &mut F,
) where
    F: FnMut(&AgentEvent),
{
    if matches!(
        &event,
        CurrentTurnHostEvent::ToolLifecycle(AgentEvent::ItemCompleted { .. })
    ) {
        drain_ready_agent_events(agent_event_receiver, on_event);
    }
    handle_host_event(
        event,
        provider_name,
        policy,
        artifact_events,
        web_search_tracker,
        usage,
        on_event,
    );
}

fn drain_ready_turn_events<F>(
    host_event_receiver: &mut mpsc::UnboundedReceiver<CurrentTurnHostEvent>,
    agent_event_receiver: &mut mpsc::UnboundedReceiver<AgentEvent>,
    provider_name: &str,
    policy: &RequestToolPolicy,
    artifact_events: &mut WriteArtifactEventEmitter,
    web_search_tracker: &mut WebSearchExecutionTracker,
    usage: &mut Option<model_provider::current_client::CurrentProviderUsage>,
    on_event: &mut F,
) where
    F: FnMut(&AgentEvent),
{
    while let Ok(event) = host_event_receiver.try_recv() {
        handle_ordered_host_event(
            event,
            agent_event_receiver,
            provider_name,
            policy,
            artifact_events,
            web_search_tracker,
            usage,
            on_event,
        );
    }
    drain_ready_agent_events(agent_event_receiver, on_event);
}

fn drain_ready_agent_events<F>(
    agent_event_receiver: &mut mpsc::UnboundedReceiver<AgentEvent>,
    on_event: &mut F,
) where
    F: FnMut(&AgentEvent),
{
    while let Ok(event) = agent_event_receiver.try_recv() {
        on_event(&event);
    }
}

fn handle_host_event<F>(
    event: CurrentTurnHostEvent,
    provider_name: &str,
    policy: &RequestToolPolicy,
    artifact_events: &mut WriteArtifactEventEmitter,
    web_search_tracker: &mut WebSearchExecutionTracker,
    usage: &mut Option<model_provider::current_client::CurrentProviderUsage>,
    on_event: &mut F,
) where
    F: FnMut(&AgentEvent),
{
    match event {
        CurrentTurnHostEvent::Provider(event) => {
            handle_provider_event(event, provider_name, artifact_events, usage, on_event)
        }
        CurrentTurnHostEvent::ToolLifecycle(event) => {
            match &event {
                AgentEvent::ItemStarted { item } => {
                    if let ThreadItemPayload::Tool { call_id, name, .. } = &item.payload {
                        web_search_tracker.record_tool_start(policy, call_id, name);
                    }
                }
                AgentEvent::ItemCompleted { item } => {
                    if let ThreadItemPayload::Tool {
                        call_id, output, ..
                    } = &item.payload
                    {
                        web_search_tracker.record_tool_end(
                            policy,
                            call_id,
                            item.status == ItemStatus::Completed,
                            output.as_ref().and_then(|output| output.error.as_deref()),
                        );
                    }
                }
                _ => {}
            }
            emit_with_artifacts(artifact_events, event, on_event);
        }
    }
}

fn handle_provider_event<F>(
    event: CurrentProviderTurnEvent,
    _provider_name: &str,
    artifact_events: &mut WriteArtifactEventEmitter,
    usage: &mut Option<model_provider::current_client::CurrentProviderUsage>,
    on_event: &mut F,
) where
    F: FnMut(&AgentEvent),
{
    match event {
        CurrentProviderTurnEvent::ProviderTrace { event } => emit_with_artifacts(
            artifact_events,
            AgentEvent::ProviderTrace { event },
            on_event,
        ),
        CurrentProviderTurnEvent::TextStart { item_id } => {
            emit_with_artifacts(artifact_events, AgentEvent::TextStart { item_id }, on_event)
        }
        CurrentProviderTurnEvent::TextDelta { item_id, text } => emit_with_artifacts(
            artifact_events,
            AgentEvent::TextDelta { item_id, text },
            on_event,
        ),
        CurrentProviderTurnEvent::TextEnd { item_id, phase } => emit_with_artifacts(
            artifact_events,
            AgentEvent::TextEnd {
                item_id,
                phase: match phase {
                    agent_runtime::provider_turn::CurrentProviderTextPhase::Commentary => {
                        crate::protocol::AgentMessagePhase::Commentary
                    }
                    agent_runtime::provider_turn::CurrentProviderTextPhase::FinalAnswer => {
                        crate::protocol::AgentMessagePhase::FinalAnswer
                    }
                },
            },
            on_event,
        ),
        CurrentProviderTurnEvent::ReasoningStart { item_id } => emit_with_artifacts(
            artifact_events,
            AgentEvent::ReasoningStart { item_id },
            on_event,
        ),
        CurrentProviderTurnEvent::ReasoningSummaryDelta {
            item_id,
            text,
            summary_index,
        } => emit_with_artifacts(
            artifact_events,
            AgentEvent::ReasoningSummaryDelta {
                item_id,
                text,
                summary_index,
            },
            on_event,
        ),
        CurrentProviderTurnEvent::ReasoningSummaryPartAdded {
            item_id,
            summary_index,
        } => emit_with_artifacts(
            artifact_events,
            AgentEvent::ReasoningSummaryPartAdded {
                item_id,
                summary_index,
            },
            on_event,
        ),
        CurrentProviderTurnEvent::ReasoningContentDelta {
            item_id,
            text,
            content_index,
        } => emit_with_artifacts(
            artifact_events,
            AgentEvent::ReasoningContentDelta {
                item_id,
                text,
                content_index,
            },
            on_event,
        ),
        CurrentProviderTurnEvent::ReasoningEnd { item_id } => emit_with_artifacts(
            artifact_events,
            AgentEvent::ReasoningEnd { item_id },
            on_event,
        ),
        CurrentProviderTurnEvent::ToolInputDelta {
            tool_id: _,
            tool_name: _,
            delta: _,
            accumulated_arguments: _,
        } => {}
        CurrentProviderTurnEvent::Usage { attempt, usage } => emit_with_artifacts(
            artifact_events,
            AgentEvent::ProviderUsage {
                attempt,
                usage: project_usage(usage),
            },
            on_event,
        ),
        CurrentProviderTurnEvent::ServerModel { model } => {
            emit_with_artifacts(artifact_events, AgentEvent::ServerModel { model }, on_event)
        }
        CurrentProviderTurnEvent::ModelReroute {
            from_model,
            to_model,
            reason,
        } => emit_with_artifacts(
            artifact_events,
            AgentEvent::ModelReroute {
                from_model,
                to_model,
                reason,
            },
            on_event,
        ),
        CurrentProviderTurnEvent::ModelVerification { verifications } => emit_with_artifacts(
            artifact_events,
            AgentEvent::ModelVerification { verifications },
            on_event,
        ),
        CurrentProviderTurnEvent::ProviderStep {
            attempt,
            completed,
            finish_reason,
            text_output_chars,
            reasoning_output_chars,
            tool_call_count,
            usage: step_usage,
        } => {
            if let Some(step_usage) = step_usage.as_ref() {
                accumulate_usage(usage, step_usage);
            }
            emit_with_artifacts(
                artifact_events,
                AgentEvent::ProviderStep {
                    attempt,
                    completed,
                    finish_reason,
                    text_output_chars,
                    reasoning_output_chars,
                    tool_call_count,
                    usage: step_usage.map(project_usage),
                },
                on_event,
            );
        }
    }
}

fn emit_with_artifacts<F>(
    artifact_events: &mut WriteArtifactEventEmitter,
    mut event: AgentEvent,
    on_event: &mut F,
) where
    F: FnMut(&AgentEvent),
{
    for extra in artifact_events.process_event(&mut event) {
        on_event(&extra);
    }
    on_event(&event);
}

fn is_web_tool(name: &str) -> bool {
    is_same_tool(name, "WebSearch") || is_same_tool(name, "WebFetch")
}

fn project_usage(usage: model_provider::current_client::CurrentProviderUsage) -> AgentTokenUsage {
    AgentTokenUsage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cached_input_tokens: usage.cached_input_tokens,
        cache_creation_input_tokens: usage.cache_creation_input_tokens,
    }
}

fn accumulate_usage(
    total: &mut Option<model_provider::current_client::CurrentProviderUsage>,
    step: &model_provider::current_client::CurrentProviderUsage,
) {
    let total = total.get_or_insert_default();
    total.input_tokens = total.input_tokens.saturating_add(step.input_tokens);
    total.output_tokens = total.output_tokens.saturating_add(step.output_tokens);
    total.cached_input_tokens =
        sum_optional_tokens(total.cached_input_tokens, step.cached_input_tokens);
    total.cache_creation_input_tokens = sum_optional_tokens(
        total.cache_creation_input_tokens,
        step.cache_creation_input_tokens,
    );
}

fn sum_optional_tokens(left: Option<u32>, right: Option<u32>) -> Option<u32> {
    match (left, right) {
        (None, None) => None,
        (left, right) => Some(
            left.unwrap_or_default()
                .saturating_add(right.unwrap_or_default()),
        ),
    }
}

fn default_working_directory() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use rmcp::model::{CallToolResult, Content, JsonObject, ListToolsResult};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::Mutex;
    use tool_runtime::mcp_connection::{McpConnection, McpConnectionCall, McpConnectionError};
    use tool_runtime::tool_call::ToolEnvironment;
    use tool_runtime::tool_executor::{RuntimeToolExecutionRequest, RuntimeToolPolicyErrorKind};
    use tool_runtime::tool_extension::RuntimeExtensionConfig;
    use tool_runtime::tool_io::{
        ToolIoPayloadStats, ToolOutputReference, ToolOutputTruncation, ToolOutputTruncationReason,
    };
    use tool_runtime::tool_result_projection::NormalizedToolOutput;

    struct HangingMcpConnection;

    #[async_trait]
    impl McpConnection for HangingMcpConnection {
        async fn list_tools(
            &self,
            _next_cursor: Option<String>,
            _cancel_token: CancellationToken,
        ) -> Result<ListToolsResult, McpConnectionError> {
            std::future::pending().await
        }

        async fn start_call_tool(
            &self,
            _name: &str,
            _arguments: Option<JsonObject>,
            _scope: &tool_runtime::mcp_connection::McpCallScope,
            _cancel_token: CancellationToken,
        ) -> Result<McpConnectionCall, McpConnectionError> {
            Ok(McpConnectionCall {
                response: Box::pin(std::future::pending()),
                notifications: Box::pin(futures::stream::empty()),
            })
        }
    }

    #[test]
    fn tool_result_projection_keeps_text_and_structured_content() {
        let result = project_call_result(Ok(CallToolResult {
            content: vec![Content::text("workspace result")],
            structured_content: Some(serde_json::json!({ "path": "README.md" })),
            is_error: Some(false),
            meta: None,
        }))
        .expect("tool result");

        assert!(result.success);
        assert_eq!(result.output, "workspace result");
        assert_eq!(
            result.structured_content,
            Some(serde_json::json!({ "path": "README.md" }))
        );
        assert!(!result.metadata.contains_key("structured_content"));
    }

    #[test]
    fn lifecycle_projection_preserves_normalized_terminal_output() {
        let (sender, _receiver) = mpsc::unbounded_channel();
        let emitter = CurrentTurnToolLifecycleEmitter::new(sender, "session-1", "thread-1");
        let started = emitter
            .project(ToolLifecycleEvent {
                turn_id: "turn-1".to_string(),
                call_id: "call-1".to_string(),
                tool_name: "Read".to_string(),
                arguments: serde_json::json!({ "path": "README.md" }),
                provider_metadata: serde_json::json!({
                    "google": { "thoughtSignature": "sig" }
                }),
                environments: vec![ToolEnvironment::new("local", PathBuf::from("/workspace"))],
                phase: ToolLifecyclePhase::Started,
                output: None,
            })
            .expect("started lifecycle projection");
        let AgentEvent::ItemStarted { item: started } = started else {
            panic!("expected canonical item started event");
        };
        assert_eq!(started.session_id.as_str(), "session-1");
        assert_eq!(started.thread_id.as_str(), "thread-1");
        assert_eq!(started.turn_id.as_str(), "turn-1");
        assert_eq!(started.item_id.as_str(), "item_call-1");
        assert_eq!(started.status, ItemStatus::InProgress);
        let ThreadItemPayload::Tool {
            call_id,
            name,
            arguments,
            output,
        } = &started.payload
        else {
            panic!("expected canonical tool payload");
        };
        assert_eq!(call_id, "call-1");
        assert_eq!(name, "Read");
        assert_eq!(arguments[0].name, "path");
        assert_eq!(arguments[0].value, "README.md");
        assert!(output.is_none());
        assert_eq!(
            started.metadata["provider_metadata"]["google"]["thoughtSignature"],
            "sig"
        );

        let projected = emitter
            .project(ToolLifecycleEvent {
                turn_id: "turn-1".to_string(),
                call_id: "call-1".to_string(),
                tool_name: "Read".to_string(),
                arguments: serde_json::json!({ "path": "README.md" }),
                provider_metadata: serde_json::json!({
                    "google": { "thoughtSignature": "sig" }
                }),
                environments: vec![ToolEnvironment::new("local", PathBuf::from("/workspace"))],
                phase: ToolLifecyclePhase::Completed,
                output: Some(NormalizedToolOutput {
                    success: true,
                    text: "preview".to_string(),
                    structured_content: Some(serde_json::json!({ "rows": 3 })),
                    error: None,
                    duration_ms: 42,
                    truncation: Some(ToolOutputTruncation::new(
                        ToolOutputTruncationReason::PayloadOffloaded,
                        ToolIoPayloadStats {
                            chars: 12_000,
                            bytes: 12_000,
                            tokens: 3_000,
                        },
                    )),
                    sidecar_reference: Some(ToolOutputReference::new(
                        "sidecar://tool-output-1",
                        Some("preview".to_string()),
                    )),
                    metadata: HashMap::from([("source".to_string(), serde_json::json!("mcp"))]),
                    agent_control_projection_facts: Vec::new(),
                    agent_control_state_facts: Vec::new(),
                }),
            })
            .expect("completed lifecycle projection");

        let AgentEvent::ItemCompleted { item } = projected else {
            panic!("expected canonical item completed event");
        };
        assert_eq!(item.session_id.as_str(), "session-1");
        assert_eq!(item.thread_id.as_str(), "thread-1");
        assert_eq!(item.turn_id.as_str(), "turn-1");
        assert_eq!(item.item_id, started.item_id);
        assert_eq!(item.ordinal, started.ordinal);
        assert_eq!(item.created_at_ms, started.created_at_ms);
        assert_eq!(item.status, ItemStatus::Completed);
        let ThreadItemPayload::Tool {
            call_id,
            name,
            arguments,
            output,
        } = item.payload
        else {
            panic!("expected canonical tool payload");
        };
        assert_eq!(call_id, "call-1");
        assert_eq!(name, "Read");
        assert_eq!(arguments[0].value, "README.md");
        let output = output.expect("canonical terminal output");
        assert_eq!(output.text.as_deref(), Some("preview"));
        assert_eq!(
            output.structured_content,
            Some(serde_json::json!({ "rows": 3 }))
        );
        assert_eq!(output.duration_ms, Some(42));
        assert!(output.truncated);
        assert_eq!(
            output.output_ref.as_deref(),
            Some("sidecar://tool-output-1")
        );
        let metadata = item.metadata.as_object().expect("canonical item metadata");
        assert_eq!(metadata.get("source"), Some(&serde_json::json!("mcp")));
        assert_eq!(metadata.get("duration_ms"), Some(&serde_json::json!(42)));
        assert_eq!(
            metadata
                .get("truncation")
                .and_then(|value| value.get("reason")),
            Some(&serde_json::json!("payload_offloaded"))
        );
        assert_eq!(
            metadata
                .get("sidecar_reference")
                .and_then(|value| value.get("reference")),
            Some(&serde_json::json!("sidecar://tool-output-1"))
        );
        assert_eq!(metadata["environments"][0]["environmentId"], "local");
        assert_eq!(metadata["environments"][0]["cwd"], "/workspace");
        assert_eq!(
            metadata["provider_metadata"]["google"]["thoughtSignature"],
            "sig"
        );
    }

    #[test]
    fn lifecycle_projection_maps_failed_output_to_failed_item() {
        let (sender, _receiver) = mpsc::unbounded_channel();
        let emitter = CurrentTurnToolLifecycleEmitter::new(sender, "session-1", "thread-1");
        let output_metadata = HashMap::from([
            ("command".to_string(), serde_json::json!("false")),
            ("command_output".to_string(), serde_json::json!("failed")),
            ("exit_code".to_string(), serde_json::json!(1)),
        ]);
        let projected = emitter
            .project(ToolLifecycleEvent {
                turn_id: "turn-1".to_string(),
                call_id: "call-failed".to_string(),
                tool_name: "exec_command".to_string(),
                arguments: serde_json::json!({ "command": "false" }),
                provider_metadata: serde_json::Value::Null,
                environments: Vec::new(),
                phase: ToolLifecyclePhase::Completed,
                output: Some(NormalizedToolOutput {
                    success: false,
                    text: "failed".to_string(),
                    structured_content: None,
                    error: Some("exit code 1".to_string()),
                    duration_ms: 3,
                    truncation: None,
                    sidecar_reference: None,
                    metadata: output_metadata,
                    agent_control_projection_facts: Vec::new(),
                    agent_control_state_facts: Vec::new(),
                }),
            })
            .expect("failed lifecycle projection");

        let AgentEvent::ItemCompleted { item } = projected else {
            panic!("expected canonical item completed event");
        };
        assert_eq!(item.status, ItemStatus::Failed);
        let ThreadItemPayload::Command {
            command,
            output,
            exit_code,
            ..
        } = item.payload
        else {
            panic!("expected canonical command payload");
        };
        assert_eq!(command, "false");
        assert_eq!(output.as_deref(), Some("failed"));
        assert_eq!(exit_code, Some(1));
    }

    #[tokio::test]
    async fn ready_lifecycle_and_policy_events_keep_start_action_end_order() {
        let (host_sender, mut host_receiver) = mpsc::unbounded_channel();
        let (agent_sender, mut agent_receiver) = mpsc::unbounded_channel();
        let emitter = CurrentTurnToolLifecycleEmitter::new(host_sender, "session-1", "thread-1");
        let environment = vec![ToolEnvironment::new("local", PathBuf::from("/workspace"))];

        emitter
            .emit(ToolLifecycleEvent {
                turn_id: "turn-1".to_string(),
                call_id: "call-1".to_string(),
                tool_name: "Read".to_string(),
                arguments: serde_json::json!({ "path": "README.md" }),
                provider_metadata: serde_json::Value::Null,
                environments: environment.clone(),
                phase: ToolLifecyclePhase::Started,
                output: None,
            })
            .await;
        agent_sender
            .send(AgentEvent::ActionRequired {
                request_id: "approval-1".to_string(),
                action_type: "tool_confirmation".to_string(),
                data: serde_json::json!({ "toolCallId": "call-1" }),
                scope: None,
            })
            .expect("queue policy event");
        emitter
            .emit(ToolLifecycleEvent {
                turn_id: "turn-1".to_string(),
                call_id: "call-1".to_string(),
                tool_name: "Read".to_string(),
                arguments: serde_json::json!({ "path": "README.md" }),
                provider_metadata: serde_json::Value::Null,
                environments: environment,
                phase: ToolLifecyclePhase::Completed,
                output: Some(NormalizedToolOutput {
                    success: true,
                    text: "done".to_string(),
                    structured_content: None,
                    error: None,
                    duration_ms: 1,
                    truncation: None,
                    sidecar_reference: None,
                    metadata: HashMap::new(),
                    agent_control_projection_facts: Vec::new(),
                    agent_control_state_facts: Vec::new(),
                }),
            })
            .await;

        let policy = RequestToolPolicy {
            search_mode: crate::request_tool_policy::RequestToolPolicyMode::Disabled,
            effective_web_search: false,
            required_tools: Vec::new(),
            allowed_tools: Vec::new(),
            disallowed_tools: Vec::new(),
        };
        let mut artifacts = WriteArtifactEventEmitter::new("session-1".to_string());
        let mut tracker = WebSearchExecutionTracker::default();
        let mut usage = None;
        let mut order = Vec::new();
        drain_ready_turn_events(
            &mut host_receiver,
            &mut agent_receiver,
            "provider",
            &policy,
            &mut artifacts,
            &mut tracker,
            &mut usage,
            &mut |event| match event {
                AgentEvent::ItemStarted { .. } => order.push("item.started"),
                AgentEvent::ActionRequired { .. } => order.push("action.required"),
                AgentEvent::ItemCompleted { .. } => order.push("item.completed"),
                _ => {}
            },
        );

        assert_eq!(order, ["item.started", "action.required", "item.completed"]);
    }

    #[test]
    fn request_policy_hides_disallowed_tools() {
        let policy = RequestToolPolicy {
            search_mode: crate::request_tool_policy::RequestToolPolicyMode::Disabled,
            effective_web_search: false,
            required_tools: Vec::new(),
            allowed_tools: Vec::new(),
            disallowed_tools: vec!["WebSearch".to_string()],
        };

        assert!(policy.matches_any_disallowed_tool("web_search"));
    }

    #[test]
    fn action_scope_uses_typed_call_identity_and_canonical_thread() {
        let context = tool_runtime::tool_executor::RuntimeToolExecutionContext::new(
            tool_runtime::tool_executor::RuntimeToolExecutionContextInput {
                working_directory: PathBuf::from("/workspace"),
                session_id: "session-canonical".to_string(),
                cancel_token: None,
                workspace_sandbox: None,
            },
        )
        .with_tool_identity(
            tool_runtime::tool_executor::RuntimeToolExecutionIdentity::new(
                "call-canonical",
                "turn-canonical",
            ),
        );
        let turn_context = tool_runtime::tool_executor::RuntimeToolTurnContext {
            metadata: HashMap::from([
                (
                    "thread_id".to_string(),
                    serde_json::json!("thread-metadata"),
                ),
                ("turn_id".to_string(), serde_json::json!("turn-metadata")),
                (
                    "tool_call_id".to_string(),
                    serde_json::json!("call-metadata"),
                ),
            ]),
            ..tool_runtime::tool_executor::RuntimeToolTurnContext::default()
        };
        let params = serde_json::json!({});

        let (scope, call_id) = action_scope(
            RuntimeToolExecutionRequest {
                tool_name: "exec_command",
                params: &params,
                context: &context,
                turn_context: Some(&turn_context),
            },
            &ThreadId::new("thread-canonical"),
        )
        .expect("typed tool identity should form approval scope");

        let scope = scope.expect("canonical action scope");
        assert_eq!(scope.session_id.as_deref(), Some("session-canonical"));
        assert_eq!(scope.thread_id.as_deref(), Some("thread-canonical"));
        assert_ne!(scope.thread_id, scope.session_id);
        assert_eq!(scope.turn_id.as_deref(), Some("turn-canonical"));
        assert_eq!(call_id, "call-canonical");
    }

    #[test]
    fn action_scope_fails_closed_without_typed_call_identity() {
        let context = tool_runtime::tool_executor::RuntimeToolExecutionContext::new(
            tool_runtime::tool_executor::RuntimeToolExecutionContextInput {
                working_directory: PathBuf::from("/workspace"),
                session_id: "session-1".to_string(),
                cancel_token: None,
                workspace_sandbox: None,
            },
        );
        let turn_context = tool_runtime::tool_executor::RuntimeToolTurnContext {
            metadata: HashMap::from([
                (
                    "thread_id".to_string(),
                    serde_json::json!("thread-metadata"),
                ),
                ("turn_id".to_string(), serde_json::json!("turn-metadata")),
                (
                    "tool_call_id".to_string(),
                    serde_json::json!("call-metadata"),
                ),
            ]),
            ..tool_runtime::tool_executor::RuntimeToolTurnContext::default()
        };
        let params = serde_json::json!({});

        let error = action_scope(
            RuntimeToolExecutionRequest {
                tool_name: "exec_command",
                params: &params,
                context: &context,
                turn_context: Some(&turn_context),
            },
            &ThreadId::new("thread-1"),
        )
        .expect_err("metadata identity must not revive an unidentified tool request");

        assert_eq!(
            error.message(),
            "tool approval requires canonical tool identity"
        );
        assert!(matches!(
            error.policy_kind(),
            Some(RuntimeToolPolicyErrorKind::ExecutionFailed(reason))
                if reason == "tool_approval_identity_missing"
        ));
    }

    #[test]
    fn mcp_call_scope_uses_canonical_execution_identity() {
        let context = tool_runtime::tool_executor::RuntimeToolExecutionContext::new(
            tool_runtime::tool_executor::RuntimeToolExecutionContextInput {
                working_directory: PathBuf::from("/workspace"),
                session_id: "session-canonical".to_string(),
                cancel_token: None,
                workspace_sandbox: None,
            },
        )
        .with_tool_identity(
            tool_runtime::tool_executor::RuntimeToolExecutionIdentity::new(
                "call-canonical",
                "turn-canonical",
            ),
        );
        let turn_context = tool_runtime::tool_executor::RuntimeToolTurnContext {
            metadata: HashMap::from([
                (
                    "thread_id".to_string(),
                    serde_json::json!("thread-metadata"),
                ),
                ("turn_id".to_string(), serde_json::json!("turn-metadata")),
                (
                    "tool_call_id".to_string(),
                    serde_json::json!("call-metadata"),
                ),
            ]),
            ..tool_runtime::tool_executor::RuntimeToolTurnContext::default()
        };
        let params = serde_json::json!({});

        let scope = mcp_call_scope(RuntimeToolExecutionRequest {
            tool_name: "mcp_search",
            params: &params,
            context: &context,
            turn_context: Some(&turn_context),
        })
        .expect("canonical execution identity should form MCP call scope");

        assert_eq!(scope.turn_id(), Some("turn-canonical"));
    }

    #[test]
    fn mcp_call_scope_fails_closed_without_typed_call_identity() {
        let context = tool_runtime::tool_executor::RuntimeToolExecutionContext::new(
            tool_runtime::tool_executor::RuntimeToolExecutionContextInput {
                working_directory: PathBuf::from("/workspace"),
                session_id: "session-1".to_string(),
                cancel_token: None,
                workspace_sandbox: None,
            },
        );
        let turn_context = tool_runtime::tool_executor::RuntimeToolTurnContext {
            metadata: HashMap::from([
                ("turn_id".to_string(), serde_json::json!("turn-metadata")),
                (
                    "tool_call_id".to_string(),
                    serde_json::json!("call-metadata"),
                ),
            ]),
            ..tool_runtime::tool_executor::RuntimeToolTurnContext::default()
        };
        let params = serde_json::json!({});

        let error = mcp_call_scope(RuntimeToolExecutionRequest {
            tool_name: "mcp_search",
            params: &params,
            context: &context,
            turn_context: Some(&turn_context),
        })
        .expect_err("metadata identity must not revive an unidentified MCP call");

        assert_eq!(error.message(), "MCP call requires canonical tool identity");
        assert!(matches!(
            error.policy_kind(),
            Some(RuntimeToolPolicyErrorKind::ExecutionFailed(reason))
                if reason == "mcp_call_scope_missing"
        ));
    }

    #[tokio::test]
    async fn mcp_tool_discovery_timeout_keeps_main_turn_unblocked() {
        let state = AgentRuntimeState::new();
        let runtime = Arc::new(crate::runtime_state::McpThreadRuntime::for_test(
            "session-1",
            "thread-1",
        ));
        runtime
            .connections()
            .register(
                "slow".to_string(),
                RuntimeExtensionConfig::new(
                    "slow",
                    "slow MCP fixture",
                    vec!["search".to_string()],
                    false,
                    vec!["search".to_string()],
                    None,
                ),
                tool_runtime::mcp_connection::McpConnectionProvenance::default(),
                false,
                Arc::new(Mutex::new(Box::new(HangingMcpConnection))),
            )
            .await;

        let snapshot = tokio::time::timeout(
            Duration::from_millis(200),
            mcp_step_snapshot::mcp_step_snapshot(
                &state,
                "session-1",
                &ThreadId::new("thread-1"),
                Duration::from_millis(10),
                &mcp_step_snapshot::DeferredToolSelections::default(),
            ),
        )
        .await
        .expect("MCP discovery timeout should return control to the main turn");

        assert!(snapshot.tools().is_empty());
        assert_eq!(runtime.connections().names().await, vec!["slow"]);
    }
}
