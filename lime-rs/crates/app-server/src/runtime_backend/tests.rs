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
}

impl RuntimeEventSink for TestRuntimeEventSink {
    fn emit(&mut self, event: RuntimeEvent) -> Result<(), RuntimeCoreError> {
        self.events.push(event);
        Ok(())
    }
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
