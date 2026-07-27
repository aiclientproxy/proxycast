use super::request_context::{
    apply_app_server_turn_policy, effective_runtime_options_for_turn, host_reasoning_effort,
    host_thinking_enabled, request_workspace_scope, resolve_runtime_model_selection,
    selection_from_explicit_preferences, selection_from_host_provider_config,
    selection_from_session_default, selection_with_capability_reasoning,
    should_use_compact_tool_surface, turn_context_from_request, RuntimeModelSelection,
};
use super::*;
use crate::runtime::ToolInventoryReadRequest;
use crate::NoopAppDataSource;
use crate::RuntimeHostContext;
use crate::{ActionRespondRequest, CancelExecutionRequest, ExecutionBackend};
use agent_protocol::turn_context::TurnOutputSchemaSource;
use app_server_protocol::AgentInput;
use app_server_protocol::AgentSession;
use app_server_protocol::AgentSessionActionScope;
use app_server_protocol::AgentSessionStatus;
use app_server_protocol::AgentTurn;
use app_server_protocol::AgentTurnStatus;
use app_server_protocol::BusinessObjectRef;
use app_server_protocol::RuntimeOptions;
use lime_agent::agent_tools::catalog::{
    MEMORY_ADD_NOTE_TOOL_NAME, MEMORY_LIST_TOOL_NAME, MEMORY_READ_TOOL_NAME,
    MEMORY_SEARCH_TOOL_NAME, TOOL_SEARCH_TOOL_NAME,
};
use lime_agent::{
    AgentEvent as RuntimeAgentEvent, AgentToolResult, RequestToolPolicyMode, SessionProviderConfig,
};
use serde_json::Value;
use std::collections::HashMap;
use tempfile::TempDir;

mod coding_event_projection;
mod image_tools;
mod model_selection;
mod session_prompt_context;
mod session_skill_context;
mod session_soul_context;
mod tool_inventory;
mod tool_policy_context;
mod tool_surface;
mod turn_flows;
mod workspace_scope_context;

#[derive(Default)]
struct TestRuntimeEventSink {
    events: Vec<RuntimeEvent>,
    transient_events: Vec<RuntimeEvent>,
}

impl RuntimeEventSink for TestRuntimeEventSink {
    fn emit(&mut self, event: RuntimeEvent) -> Result<(), RuntimeCoreError> {
        self.events.push(event);
        Ok(())
    }

    fn emit_transient(&mut self, event: RuntimeEvent) -> Result<(), RuntimeCoreError> {
        self.transient_events.push(event);
        Ok(())
    }
}

#[test]
fn model_events_preserve_route_evidence_and_reroute_is_transient() {
    let mut sink = TestRuntimeEventSink::default();
    let mut coding_event_mirror = coding_events::CodingEventMirror::default();
    let mut proposed_plan_parser = proposed_plan_parser::ProposedPlanParser::default();
    let mut reasoning_event_state = reasoning_events::ReasoningEventState::default();
    let evidence = ModelRouteEvidence {
        provider: "openai".to_string(),
        requested_model: "gpt-5-codex".to_string(),
        selected_model: "gpt-5-codex".to_string(),
        route_attempt: 2,
    };

    emit_runtime_agent_event_with_coding_mirror_and_plan_parser_with_soul_style(
        &RuntimeAgentEvent::ServerModel {
            model: "gpt-5.1-codex".to_string(),
        },
        &mut sink,
        &mut coding_event_mirror,
        &mut proposed_plan_parser,
        &mut reasoning_event_state,
        None,
        Some(&evidence),
    )
    .expect("server model evidence");
    emit_runtime_agent_event_with_coding_mirror_and_plan_parser_with_soul_style(
        &RuntimeAgentEvent::ModelReroute {
            from_model: "gpt-5-codex".to_string(),
            to_model: "gpt-5.1-codex".to_string(),
            reason: model_provider::current_client::ModelRerouteReason::HighRiskCyberActivity,
        },
        &mut sink,
        &mut coding_event_mirror,
        &mut proposed_plan_parser,
        &mut reasoning_event_state,
        None,
        Some(&evidence),
    )
    .expect("model reroute notification event");

    assert_eq!(sink.events.len(), 1);
    assert_eq!(sink.events[0].event_type, "model.server_reported");
    assert_eq!(sink.events[0].payload["provider"], "openai");
    assert_eq!(sink.events[0].payload["requestedModel"], "gpt-5-codex");
    assert_eq!(sink.events[0].payload["selectedModel"], "gpt-5-codex");
    assert_eq!(sink.events[0].payload["routeAttempt"], 2);
    assert_eq!(sink.transient_events.len(), 1);
    assert_eq!(sink.transient_events[0].event_type, "model.rerouted");
}

#[test]
fn reply_attempt_usage_limit_maps_to_structured_runtime_error() {
    let error = runtime_error_from_reply_attempt(
        lime_agent::ReplyAttemptError::usage_limit_exceeded("provider quota exhausted", true),
    );

    assert!(matches!(
        error,
        RuntimeCoreError::UsageLimitExceeded(message) if message == "provider quota exhausted"
    ));
}

#[test]
fn runtime_reroute_policy_rejects_direct_hard_and_partial_failures() {
    let selection = RuntimeModelSelection {
        provider: "primary-provider".to_string(),
        model: "primary-model".to_string(),
        source: "profile_model_slot",
        reasoning_effort: None,
    };
    let transport = lime_agent::ReplyAttemptError::provider_failure(
        "transport failed",
        false,
        Some(runtime_core::FailureClassification::Transport),
        true,
    );
    let exclusion = runtime_route_exclusion(&selection, false, &transport)
        .expect("retryable untouched profile route");
    assert_eq!(exclusion.provider, "primary-provider");
    assert_eq!(exclusion.model, "primary-model");
    assert!(runtime_route_exclusion(&selection, true, &transport).is_none());

    for error in [
        lime_agent::ReplyAttemptError::provider_failure(
            "authentication failed",
            false,
            Some(runtime_core::FailureClassification::Authentication),
            false,
        ),
        lime_agent::ReplyAttemptError::provider_failure(
            "permission denied",
            false,
            Some(runtime_core::FailureClassification::Permission),
            false,
        ),
        lime_agent::ReplyAttemptError::provider_failure(
            "quota exhausted",
            false,
            Some(runtime_core::FailureClassification::Quota),
            false,
        ),
        lime_agent::ReplyAttemptError::provider_failure(
            "partial output failed",
            true,
            Some(runtime_core::FailureClassification::Transport),
            true,
        ),
    ] {
        assert!(runtime_route_exclusion(&selection, false, &error).is_none());
    }
}

pub(super) fn request_for_test(
    message: &str,
    runtime_request: Option<app_server_protocol::RuntimeRequest>,
    metadata: Option<Value>,
) -> ExecutionRequest {
    let runtime_request = match (runtime_request, metadata) {
        (Some(mut runtime_request), Some(metadata)) => {
            runtime_request.metadata = Some(metadata);
            Some(runtime_request)
        }
        (Some(runtime_request), None) => Some(runtime_request),
        (None, Some(metadata)) => Some(app_server_protocol::RuntimeRequest {
            metadata: Some(metadata),
            ..app_server_protocol::RuntimeRequest::default()
        }),
        (None, None) => None,
    };
    ExecutionRequest {
        host: RuntimeHostContext::default(),
        session: AgentSession {
            session_id: "session-1".to_string(),
            thread_id: "thread-1".to_string(),
            app_id: "content-studio".to_string(),
            workspace_id: Some("workspace-main".to_string()),
            business_object_ref: None,
            status: AgentSessionStatus::Running,
            created_at: "2026-06-07T00:00:00.000Z".to_string(),
            updated_at: "2026-06-07T00:00:00.000Z".to_string(),
        },
        turn: AgentTurn {
            turn_id: "turn-1".to_string(),
            session_id: "session-1".to_string(),
            thread_id: "thread-1".to_string(),
            status: AgentTurnStatus::Accepted,
            started_at: None,
            completed_at: None,
        },
        forked_from_thread_id: None,
        input: agent_runtime::reply_input::RuntimeReplyInput::text(message),
        runtime_options: Some(RuntimeOptions {
            stream: true,
            runtime_request,
            ..RuntimeOptions::default()
        }),
        event_name: None,
        expected_output: None,
        structured_output: None,
        output_schema: None,
        queued_turn_id: None,
        queue_if_busy: false,
        skip_pre_submit_resume: false,
        agent_control_gateway: None,
    }
}

pub(super) fn apply_detached_agent_chat_first_turn_policy(request: &mut ExecutionRequest) {
    request.session.app_id = "agent-chat".to_string();
    request.session.workspace_id = None;
    request.session.business_object_ref = Some(BusinessObjectRef {
        kind: "agent.thread".to_string(),
        id: request.session.thread_id.clone(),
        title: None,
        uri: None,
        metadata: None,
    });
    let host_request = super::request_context::runtime_request_from_request(request);
    let tool_policy =
        super::request_context::request_tool_policy_from_request(host_request.as_ref());
    apply_app_server_turn_policy(request, true, &tool_policy);
}

fn request_with_session_metadata(metadata: Value) -> ExecutionRequest {
    let mut request = request_for_test("hello", None, None);
    request.session.business_object_ref = Some(BusinessObjectRef {
        kind: "agent_session".to_string(),
        id: "session-1".to_string(),
        title: None,
        uri: None,
        metadata: Some(metadata),
    });
    request.runtime_options = None;
    request
}

fn imported_request_with_session_metadata(metadata: Value) -> ExecutionRequest {
    let mut request = request_with_session_metadata(metadata);
    if let Some(reference) = request.session.business_object_ref.as_mut() {
        reference.kind = "conversation.import".to_string();
    }
    request
}

fn article_workspace_snapshot_event_without_search() -> RuntimeEvent {
    RuntimeEvent::new(
        "artifact.snapshot",
        json!({
            "artifact": {
                "artifactId": "artifact-article-workspace",
                "kind": "content_factory.workspace_patch",
                "metadata": {
                    "contentFactoryWorkspacePatch": {
                        "schemaVersion": 1,
                        "appId": "content-factory-app",
                        "sessionId": "session-1",
                        "objects": [
                            {
                                "ref": {
                                    "appId": "content-factory-app",
                                    "kind": "articleDraft",
                                    "id": "article-draft-1",
                                    "sessionId": "session-1"
                                },
                                "title": "公众号文章草稿",
                                "status": "ready",
                                "source": {
                                    "taskKind": "content.article.generate",
                                    "taskId": "task-article-draft-1",
                                    "documentText": "# 草稿\n\n正文。",
                                    "finalMarkdown": "# 草稿\n\n正文。"
                                }
                            }
                        ]
                    }
                }
            }
        }),
    )
}
