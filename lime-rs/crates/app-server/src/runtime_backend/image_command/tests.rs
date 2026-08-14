use super::*;
use crate::runtime::RuntimeCoreError;
use crate::{
    AutomationManagementAppDataSource, AutomationOverviewAppDataSource, ConnectAppDataSource,
    DiagnosticsAppDataSource, GatewayAppDataSource, KnowledgeAppDataSource, McpAppDataSource,
    MediaAppDataSource, MemoryAppDataSource, ModelProviderAppDataSource, PluginDataSource,
    RightSurfaceAppDataSource, SessionAppDataSource, SkillAppDataSource, UsageStatsAppDataSource,
    VoiceAppDataSource, WorkspaceAppDataSource, WorkspaceSkillBindingAppDataSource,
};
use app_server_protocol::{
    AgentSession, AgentSessionStatus, AgentTurn, AgentTurnStatus, RuntimeOptions,
};
use async_trait::async_trait;
use std::sync::Mutex;
use tempfile::TempDir;

#[derive(Default)]
struct TestSink {
    events: Vec<RuntimeEvent>,
}

impl RuntimeEventSink for TestSink {
    fn emit(&mut self, event: RuntimeEvent) -> Result<(), RuntimeCoreError> {
        self.events.push(event);
        Ok(())
    }
}

fn canonical_tool_item(events: &[RuntimeEvent], event_type: &str) -> ThreadItem {
    let event = events
        .iter()
        .find(|event| event.event_type == event_type)
        .unwrap_or_else(|| panic!("missing {event_type}"));
    serde_json::from_value(event.payload["item"].clone())
        .unwrap_or_else(|error| panic!("invalid canonical tool item: {error}"))
}

fn tool_argument<'a>(item: &'a ThreadItem, name: &str) -> Option<&'a str> {
    let ThreadItemPayload::Tool { arguments, .. } = &item.payload else {
        return None;
    };
    arguments
        .iter()
        .find(|argument| argument.name == name)
        .map(|argument| argument.value.as_str())
}

fn assert_matching_tool_identity(started: &ThreadItem, completed: &ThreadItem) {
    assert_eq!(started.item_id, completed.item_id);
    assert_eq!(started.session_id, completed.session_id);
    assert_eq!(started.thread_id, completed.thread_id);
    assert_eq!(started.turn_id, completed.turn_id);
    let ThreadItemPayload::Tool {
        call_id: started_call_id,
        ..
    } = &started.payload
    else {
        panic!("started item must be a tool");
    };
    let ThreadItemPayload::Tool {
        call_id: completed_call_id,
        ..
    } = &completed.payload
    else {
        panic!("completed item must be a tool");
    };
    assert_eq!(started.item_id.as_str(), started_call_id);
    assert_eq!(started_call_id, completed_call_id);
}

#[derive(Default)]
struct ImageCommandTestDataSource {
    params: Mutex<Vec<MediaTaskArtifactImageCreateParams>>,
}

impl SessionAppDataSource for ImageCommandTestDataSource {}
impl WorkspaceAppDataSource for ImageCommandTestDataSource {}
impl SkillAppDataSource for ImageCommandTestDataSource {}
impl WorkspaceSkillBindingAppDataSource for ImageCommandTestDataSource {}
impl GatewayAppDataSource for ImageCommandTestDataSource {}
impl VoiceAppDataSource for ImageCommandTestDataSource {}
impl PluginDataSource for ImageCommandTestDataSource {}
impl KnowledgeAppDataSource for ImageCommandTestDataSource {}
impl AutomationOverviewAppDataSource for ImageCommandTestDataSource {}
impl McpAppDataSource for ImageCommandTestDataSource {}
impl AutomationManagementAppDataSource for ImageCommandTestDataSource {}
impl MemoryAppDataSource for ImageCommandTestDataSource {}
impl DiagnosticsAppDataSource for ImageCommandTestDataSource {}
impl UsageStatsAppDataSource for ImageCommandTestDataSource {}
impl ModelProviderAppDataSource for ImageCommandTestDataSource {}
impl ConnectAppDataSource for ImageCommandTestDataSource {}
impl RightSurfaceAppDataSource for ImageCommandTestDataSource {}

#[async_trait]
impl MediaAppDataSource for ImageCommandTestDataSource {
    async fn create_image_media_task_artifact(
        &self,
        params: MediaTaskArtifactImageCreateParams,
    ) -> Result<MediaTaskArtifactResponse, RuntimeCoreError> {
        self.params
            .lock()
            .expect("params lock")
            .push(params.clone());
        crate::media_task::create_image_generation_task_artifact(params, None)
            .map_err(RuntimeCoreError::Backend)
    }
}

#[test]
fn image_command_intent_projects_current_metadata_into_create_params() {
    let workspace = TempDir::new().expect("workspace");
    let request = request_with_metadata(json!({
        "harness": {
            "projectRoot": workspace.path().to_string_lossy(),
            "image_command_intent": {
                "kind": "image_command",
                "image_task": {
                    "prompt": "画一张广州夏天的图",
                    "mode": "generate",
                    "count": 2,
                    "provider_id": "openai",
                    "model": "gpt-image-2",
                    "executor_mode": "images_api",
                    "entry_source": "at_image_command"
                }
            }
        }
    }));
    let scope = RuntimeSessionScope {
        session_id: "session-1".to_string(),
        thread_id: "thread-1".to_string(),
        turn_id: "turn-1".to_string(),
        workspace_id: None,
    };

    let intent = parse_image_command_intent(&request, &scope)
        .expect("parse image command")
        .expect("image command intent");
    let params = intent.clone().into_create_params();

    assert_eq!(params.prompt, "画一张广州夏天的图");
    assert_eq!(params.session_id.as_deref(), Some("session-1"));
    assert_eq!(params.thread_id.as_deref(), Some("thread-1"));
    assert_eq!(params.turn_id.as_deref(), Some("turn-1"));
    assert_eq!(params.provider_id.as_deref(), Some("openai"));
    assert_eq!(params.model.as_deref(), Some("gpt-image-2"));
    assert_eq!(params.entry_source.as_deref(), Some("at_image_command"));
    assert_eq!(workflow_run_id(&intent.scope), "image-command-run-turn-1");
}

#[tokio::test]
async fn image_command_workflow_fails_closed_without_runtime_presentation() {
    let workspace = TempDir::new().expect("workspace");
    let request = request_with_metadata(json!({
        "harness": {
            "projectRoot": workspace.path().to_string_lossy(),
            "image_command_intent": {
                "kind": "image_command",
                "image_task": {
                    "prompt": "画一张广州夏天的图",
                    "mode": "generate",
                    "provider_id": "openai",
                    "model": "gpt-image-2",
                    "entry_source": "at_image_command"
                }
            }
        }
    }));
    let scope = RuntimeSessionScope {
        session_id: "session-1".to_string(),
        thread_id: "thread-1".to_string(),
        turn_id: "turn-1".to_string(),
        workspace_id: None,
    };
    let data_source = Arc::new(ImageCommandTestDataSource::default());
    let mut sink = TestSink::default();

    let handled = handle_image_command_turn_if_present(
        None,
        &request,
        &scope,
        Some(data_source.clone()),
        &mut sink,
    )
    .await
    .expect("workflow should fail closed without runtime presentation");

    assert!(handled);
    assert_eq!(data_source.params.lock().expect("params lock").len(), 0);
    assert_eq!(
        sink.events
            .iter()
            .filter(|event| !event.event_type.starts_with("workflow."))
            .map(|event| event.event_type.as_str())
            .collect::<Vec<_>>(),
        vec![
            "runtime.status",
            "item.started",
            "image_task.create_failed",
            "item.completed",
            "turn.completed"
        ]
    );
    assert!(
        sink.events
            .iter()
            .any(|event| event.event_type == "image_task.create_failed"),
        "presentation failure must block image task creation"
    );
    assert!(
        !sink
            .events
            .iter()
            .any(|event| event.event_type == "image_task.presentation.unavailable"),
        "old presentation unavailable event must stay dead"
    );
    assert_eq!(
        sink.events
            .iter()
            .filter(|event| event.event_type.starts_with("workflow."))
            .map(|event| event.event_type.as_str())
            .collect::<Vec<_>>(),
        vec![
            "workflow.run.started",
            "workflow.step.completed",
            "workflow.step.completed",
            "workflow.run.completed"
        ]
    );
    let failure_event = sink
        .events
        .iter()
        .find(|event| event.event_type == "image_task.create_failed")
        .expect("presentation failure event");
    assert_eq!(
        failure_event.payload["reasonCode"].as_str(),
        Some("image_task_presentation_runtime_unavailable")
    );
    let workflow_completed = sink
        .events
        .iter()
        .find(|event| event.event_type == "workflow.run.completed")
        .expect("workflow run completed");
    assert_eq!(
        workflow_completed.payload["run_id"].as_str(),
        Some("image-command-run-turn-1")
    );
    assert_eq!(
        workflow_completed.payload["status"].as_str(),
        Some("create_failed")
    );
    assert_eq!(
        workflow_completed.payload["redaction"]["policy"].as_str(),
        Some("workflow_audit_metadata_only")
    );
    let workflow_started = sink
        .events
        .iter()
        .find(|event| event.event_type == "workflow.run.started")
        .expect("workflow run started");
    assert_eq!(
        workflow_started.payload["steps"].as_array().unwrap().len(),
        5
    );
    assert_eq!(
        workflow_started.payload["steps"][0]["status"].as_str(),
        Some("queued")
    );
    assert_eq!(
        workflow_started.payload["steps"][2]["id"].as_str(),
        Some("create_tasks")
    );
    let create_tasks_failed = sink
        .events
        .iter()
        .find(|event| {
            event.event_type == "workflow.step.completed"
                && event.payload["stepId"].as_str() == Some("create_tasks")
        })
        .expect("create_tasks failed");
    assert_eq!(create_tasks_failed.payload["stepIndex"].as_u64(), Some(2));
    assert_eq!(create_tasks_failed.payload["stepCount"].as_u64(), Some(5));
    assert_eq!(
        create_tasks_failed.payload["stepKind"].as_str(),
        Some("tool")
    );
    assert_eq!(
        create_tasks_failed.payload["status"].as_str(),
        Some("failed")
    );
    let tool_started = canonical_tool_item(&sink.events, "item.started");
    let tool_completed = canonical_tool_item(&sink.events, "item.completed");
    assert_matching_tool_identity(&tool_started, &tool_completed);
    assert_eq!(tool_started.status, ItemStatus::InProgress);
    assert_eq!(tool_completed.status, ItemStatus::Failed);
    assert_eq!(tool_started.session_id.as_str(), "session-1");
    assert_eq!(tool_started.thread_id.as_str(), "thread-1");
    assert_eq!(tool_started.turn_id.as_str(), "turn-1");
    assert_eq!(
        tool_argument(&tool_started, "prompt"),
        Some("画一张广州夏天的图")
    );
    let ThreadItemPayload::Tool { name, output, .. } = &tool_completed.payload else {
        panic!("completed item must be a tool");
    };
    assert_eq!(name, LIME_CREATE_IMAGE_TASK_TOOL_NAME);
    assert_eq!(
        output.as_ref().and_then(|output| output.error.as_deref()),
        Some(
            "Image command presentation generation skipped because runtime backend is unavailable"
        )
    );
    assert!(output
        .as_ref()
        .and_then(|output| output.duration_ms)
        .is_some());
    assert_eq!(
        tool_completed.metadata["reasonCode"].as_str(),
        Some("image_task_presentation_runtime_unavailable")
    );
    let turn_completed = sink
        .events
        .iter()
        .find(|event| event.event_type == "turn.completed")
        .expect("turn completed");
    assert_eq!(
        turn_completed.payload["status"].as_str(),
        Some("create_failed")
    );
}

#[tokio::test]
async fn image_command_workflow_uses_existing_presentation_without_runtime_backend() {
    let workspace = TempDir::new().expect("workspace");
    let request = request_with_metadata(json!({
        "harness": {
            "projectRoot": workspace.path().to_string_lossy(),
            "image_command_intent": {
                "kind": "image_command",
                "image_task": {
                    "prompt": "画一张广州夏天的图",
                    "mode": "generate",
                    "provider_id": "openai",
                    "model": "gpt-image-2",
                    "entry_source": "at_image_command",
                    "presentation": {
                        "assistant_intro": "好啊，我来按广州夏天的明亮街景处理。",
                        "planning_summary": "用强日光、绿树和城市高楼组织画面。",
                        "completion_caption": "完成了，广州夏天的通透感和城市层次已经出来。"
                    }
                }
            }
        }
    }));
    let scope = RuntimeSessionScope {
        session_id: "session-1".to_string(),
        thread_id: "thread-1".to_string(),
        turn_id: "turn-1".to_string(),
        workspace_id: None,
    };
    let data_source = Arc::new(ImageCommandTestDataSource::default());
    let mut sink = TestSink::default();

    let handled = handle_image_command_turn_if_present(
        None,
        &request,
        &scope,
        Some(data_source.clone()),
        &mut sink,
    )
    .await
    .expect("workflow should use existing presentation");

    assert!(handled);
    assert_eq!(data_source.params.lock().expect("params lock").len(), 1);
    let event_types = sink
        .events
        .iter()
        .map(|event| event.event_type.as_str())
        .collect::<Vec<_>>();
    assert!(event_types.contains(&"reasoning.started"));
    assert!(event_types.contains(&"message.delta"));
    assert!(event_types.contains(&"image_task.presentation.generated"));
    assert!(event_types.contains(&"image_task.created"));
    assert!(event_types.contains(&"item.started"));
    assert!(event_types.contains(&"item.completed"));
    assert!(event_types.contains(&"turn.completed"));
    assert!(
        !event_types.contains(&"image_task.presentation.unavailable"),
        "old presentation unavailable event must stay dead"
    );
    assert!(
        !event_types.contains(&"image_task.create_failed"),
        "existing presentation should not fail closed before task creation"
    );
    let tool_started = canonical_tool_item(&sink.events, "item.started");
    let tool_completed = canonical_tool_item(&sink.events, "item.completed");
    assert_matching_tool_identity(&tool_started, &tool_completed);
    assert_eq!(tool_started.status, ItemStatus::InProgress);
    assert_eq!(tool_completed.status, ItemStatus::Completed);
    assert_eq!(
        tool_argument(&tool_started, "prompt"),
        Some("画一张广州夏天的图")
    );
    let ThreadItemPayload::Tool { output, .. } = &tool_completed.payload else {
        panic!("completed item must be a tool");
    };
    let output = output.as_ref().expect("completed tool output");
    assert!(output.error.is_none());
    assert!(output.duration_ms.is_some());
    assert!(output
        .text
        .as_deref()
        .is_some_and(|text| text.contains("image_generate")));
    assert!(output.structured_content.is_some());
    assert_eq!(tool_completed.metadata["success"].as_bool(), Some(true));
    assert!(tool_completed.metadata["task_id"].is_string());
    let presentation_event = sink
        .events
        .iter()
        .find(|event| event.event_type == "image_task.presentation.generated")
        .expect("presentation event");
    assert_eq!(
        presentation_event.payload["presentation"]["source"].as_str(),
        Some("metadata_provided")
    );
    let stored = data_source.params.lock().expect("params lock");
    assert_eq!(
        stored[0]
            .presentation
            .as_ref()
            .and_then(|value| value.pointer("/assistant_intro"))
            .and_then(Value::as_str),
        Some("好啊，我来按广州夏天的明亮街景处理。")
    );
}

#[tokio::test]
async fn image_command_workflow_runtime_presentation_unavailable_emits_canonical_tool_lifecycle() {
    let workspace = TempDir::new().expect("workspace");
    let request = request_with_metadata(json!({
        "harness": {
            "projectRoot": workspace.path().to_string_lossy(),
            "image_command_intent": {
                "kind": "image_command",
                "image_task": {
                    "prompt": "画一张广州夏天的图",
                    "mode": "generate",
                    "provider_id": "openai",
                    "model": "gpt-image-2",
                    "entry_source": "at_image_command"
                }
            }
        }
    }));
    let scope = RuntimeSessionScope {
        session_id: "session-1".to_string(),
        thread_id: "thread-1".to_string(),
        turn_id: "turn-1".to_string(),
        workspace_id: None,
    };
    let data_source = Arc::new(ImageCommandTestDataSource::default());
    let mut sink = TestSink::default();

    let handled = handle_image_command_turn_if_present(
        None,
        &request,
        &scope,
        Some(data_source.clone()),
        &mut sink,
    )
    .await
    .expect("workflow should fail closed when presentation runtime is unavailable");

    assert!(handled);
    assert_eq!(data_source.params.lock().expect("params lock").len(), 0);
    let event_types = sink
        .events
        .iter()
        .map(|event| event.event_type.as_str())
        .collect::<Vec<_>>();
    let tool_started_index = event_types
        .iter()
        .position(|event_type| *event_type == "item.started")
        .expect("item.started");
    let tool_completed_index = event_types
        .iter()
        .position(|event_type| *event_type == "item.completed")
        .expect("item.completed");
    assert!(
        tool_started_index < tool_completed_index,
        "terminal tool item must have a matching started item first"
    );
    assert!(event_types.contains(&"image_task.create_failed"));
    assert!(
        !event_types.contains(&"image_task.presentation.unavailable"),
        "old presentation unavailable event must stay dead"
    );
    let failure_event = sink
        .events
        .iter()
        .find(|event| event.event_type == "image_task.create_failed")
        .expect("create failed event");
    assert_eq!(
        failure_event.payload["reasonCode"].as_str(),
        Some("image_task_presentation_runtime_unavailable")
    );
    let turn_completed = sink
        .events
        .iter()
        .find(|event| event.event_type == "turn.completed")
        .expect("turn completed");
    assert_eq!(
        turn_completed.payload["status"].as_str(),
        Some("create_failed")
    );
}

#[test]
fn image_command_create_failed_emits_canonical_tool_lifecycle() {
    let workspace = TempDir::new().expect("workspace");
    let request = request_with_metadata(json!({
        "harness": {
            "projectRoot": workspace.path().to_string_lossy(),
            "image_command_intent": {
                "kind": "image_command",
                "image_task": {
                    "prompt": "画一张广州夏天的图",
                    "mode": "generate",
                    "provider_id": "openai",
                    "model": "gpt-image-2",
                    "entry_source": "at_image_command"
                }
            }
        }
    }));
    let scope = RuntimeSessionScope {
        session_id: "session-1".to_string(),
        thread_id: "thread-1".to_string(),
        turn_id: "turn-1".to_string(),
        workspace_id: None,
    };
    let intent = parse_image_command_intent(&request, &scope)
        .expect("parse image command")
        .expect("image command intent");
    let mut sink = TestSink::default();

    emit_create_failed(
        &intent,
        "image_task_create_failed",
        "fixture create failure",
        &mut sink,
    )
    .expect("create failed events");

    let event_types = sink
        .events
        .iter()
        .map(|event| event.event_type.as_str())
        .collect::<Vec<_>>();
    let tool_started_index = event_types
        .iter()
        .position(|event_type| *event_type == "item.started")
        .expect("item.started");
    let tool_completed_index = event_types
        .iter()
        .position(|event_type| *event_type == "item.completed")
        .expect("item.completed");
    assert!(
        tool_started_index < tool_completed_index,
        "terminal tool item must have a matching started item first"
    );
    assert!(event_types.contains(&"image_task.create_failed"));
    let failure_event = sink
        .events
        .iter()
        .find(|event| event.event_type == "image_task.create_failed")
        .expect("create failed event");
    assert_eq!(
        failure_event.payload["reasonCode"].as_str(),
        Some("image_task_create_failed")
    );
    let tool_started = canonical_tool_item(&sink.events, "item.started");
    let tool_completed = canonical_tool_item(&sink.events, "item.completed");
    assert_matching_tool_identity(&tool_started, &tool_completed);
    assert_eq!(tool_completed.status, ItemStatus::Failed);
    assert_eq!(
        tool_argument(&tool_completed, "prompt"),
        Some("画一张广州夏天的图")
    );
    assert_eq!(
        tool_completed.metadata["reasonCode"].as_str(),
        Some("image_task_create_failed")
    );
    let ThreadItemPayload::Tool { output, .. } = &tool_completed.payload else {
        panic!("completed item must be a tool");
    };
    assert_eq!(
        output.as_ref().and_then(|output| output.error.as_deref()),
        Some("fixture create failure")
    );
    assert!(output
        .as_ref()
        .and_then(|output| output.duration_ms)
        .is_some());
}

#[test]
fn image_command_task_created_turn_completed_includes_presentation_usage() {
    let mut sink = TestSink::default();
    let scope = RuntimeSessionScope {
        session_id: "session-usage".to_string(),
        thread_id: "thread-usage".to_string(),
        turn_id: "turn-usage".to_string(),
        workspace_id: None,
    };
    let params = MediaTaskArtifactImageCreateParams {
        prompt: "从花城汇看广州塔的春天照片".to_string(),
        ..MediaTaskArtifactImageCreateParams::default()
    };
    let usage = AgentTokenUsage {
        input_tokens: 31_000,
        output_tokens: 0,
        cached_input_tokens: Some(1_024),
        cache_creation_input_tokens: None,
        codex_rollout_budget_units: None,
    };
    let response = MediaTaskArtifactResponse {
        success: true,
        task_id: "task-image-usage".to_string(),
        task_type: "image_generate".to_string(),
        task_family: "image".to_string(),
        status: "pending_submit".to_string(),
        normalized_status: "pending".to_string(),
        artifact_path: ".lime/tasks/image_generate/task-image-usage.json".to_string(),
        record: json!({
            "task_type": "image_generate",
            "payload": {
                "prompt": "从花城汇看广州塔的春天照片"
            }
        }),
        ..MediaTaskArtifactResponse::default()
    };

    emit_task_created(
        &scope,
        &tool_call_id(&scope),
        &params,
        1,
        response.clone(),
        &mut sink,
    )
    .expect("task created events");
    emit_task_created_turn_completed(
        response.task_id.as_str(),
        response.artifact_path.as_str(),
        Some(&usage),
        &mut sink,
    )
    .expect("task created turn completed");

    let turn_completed = sink
        .events
        .iter()
        .find(|event| event.event_type == "turn.completed")
        .expect("turn completed");
    assert_eq!(
        turn_completed.payload["status"].as_str(),
        Some("task_created")
    );
    assert_eq!(
        turn_completed.payload["usage"]["input_tokens"].as_u64(),
        Some(31_000)
    );
    assert_eq!(
        turn_completed.payload["usage"]["output_tokens"].as_u64(),
        Some(0)
    );
    assert_eq!(
        turn_completed.payload["usage"]["cached_input_tokens"].as_u64(),
        Some(1_024)
    );
    assert!(turn_completed.payload["usage"]
        .get("cache_creation_input_tokens")
        .is_none());
}

#[tokio::test]
async fn image_command_workflow_ignores_retired_image_skill_launch_metadata() {
    let request = request_with_metadata(json!({
        "harness": {
            "image_skill_launch": {
                "skill_name": "image_generate",
                "kind": "image_task",
                "image_task": {
                    "prompt": "画一张广州夏天的图",
                    "mode": "generate"
                }
            }
        }
    }));
    let scope = RuntimeSessionScope {
        session_id: "session-1".to_string(),
        thread_id: "thread-1".to_string(),
        turn_id: "turn-1".to_string(),
        workspace_id: None,
    };
    let data_source = Arc::new(ImageCommandTestDataSource::default());
    let mut sink = TestSink::default();

    let handled = handle_image_command_turn_if_present(
        None,
        &request,
        &scope,
        Some(data_source.clone()),
        &mut sink,
    )
    .await
    .expect("retired image skill launch should be ignored");

    assert!(!handled);
    assert!(sink.events.is_empty());
    let stored = data_source.params.lock().expect("params lock");
    assert!(stored.is_empty());
}

#[tokio::test]
async fn image_command_workflow_requires_project_root_instead_of_using_cwd() {
    let request = request_with_metadata(json!({
        "harness": {
            "image_command_intent": {
                "kind": "image_command",
                "image_task": {
                    "prompt": "画一张广州夏天的图",
                    "mode": "generate"
                }
            }
        }
    }));
    let scope = RuntimeSessionScope {
        session_id: "session-1".to_string(),
        thread_id: "thread-1".to_string(),
        turn_id: "turn-1".to_string(),
        workspace_id: None,
    };
    let data_source = Arc::new(ImageCommandTestDataSource::default());
    let mut sink = TestSink::default();

    let handled = handle_image_command_turn_if_present(
        None,
        &request,
        &scope,
        Some(data_source.clone()),
        &mut sink,
    )
    .await
    .expect("workflow should handle missing project root");

    assert!(handled);
    assert_eq!(
        sink.events
            .iter()
            .filter(|event| !event.event_type.starts_with("workflow."))
            .map(|event| event.event_type.as_str())
            .collect::<Vec<_>>(),
        vec![
            "runtime.status",
            "image_task.parameters.required",
            "turn.completed"
        ]
    );
    assert_eq!(
        sink.events
            .iter()
            .filter(|event| event.event_type.starts_with("workflow."))
            .map(|event| event.event_type.as_str())
            .collect::<Vec<_>>(),
        vec![
            "workflow.run.started",
            "workflow.step.completed",
            "workflow.step.completed",
            "workflow.run.completed"
        ]
    );
    let workflow_started = sink
        .events
        .iter()
        .find(|event| event.event_type == "workflow.run.started")
        .expect("workflow run started");
    assert_eq!(
        workflow_started.payload["steps"]
            .as_array()
            .expect("workflow steps")
            .len(),
        5
    );
    assert_eq!(
        workflow_started.payload["steps"][2]["id"].as_str(),
        Some("create_tasks")
    );
    assert_eq!(
        sink.events
            .iter()
            .find(|event| event.event_type == "image_task.parameters.required")
            .and_then(|event| event.payload["missing"][0].as_str()),
        Some("project_root_path")
    );
    let stored = data_source.params.lock().expect("params lock");
    assert!(stored.is_empty());
}

#[tokio::test]
async fn image_command_workflow_requires_prompt_without_falling_through() {
    let request = request_with_metadata(json!({
        "harness": {
            "image_command_intent": {
                "kind": "image_command",
                "image_task": {
                    "mode": "generate"
                }
            }
        }
    }));
    let scope = RuntimeSessionScope {
        session_id: "session-1".to_string(),
        thread_id: "thread-1".to_string(),
        turn_id: "turn-1".to_string(),
        workspace_id: None,
    };
    let mut sink = TestSink::default();

    let handled = handle_image_command_turn_if_present(None, &request, &scope, None, &mut sink)
        .await
        .expect("workflow should handle missing prompt");

    assert!(handled);
    assert_eq!(
        sink.events
            .iter()
            .filter(|event| !event.event_type.starts_with("workflow."))
            .map(|event| event.event_type.as_str())
            .collect::<Vec<_>>(),
        vec![
            "runtime.status",
            "image_task.parameters.required",
            "turn.completed"
        ]
    );
    assert_eq!(
        sink.events
            .iter()
            .filter(|event| event.event_type.starts_with("workflow."))
            .map(|event| event.event_type.as_str())
            .collect::<Vec<_>>(),
        vec![
            "workflow.run.started",
            "workflow.step.completed",
            "workflow.step.completed",
            "workflow.run.completed"
        ]
    );
    assert!(
        sink.events
            .iter()
            .filter(|event| event.event_type.starts_with("workflow.step."))
            .all(|event| event.payload["stepId"].as_str() != Some("create_task")),
        "image workflow uses the current create_tasks step id"
    );
}

#[test]
fn ordinary_chat_without_image_metadata_does_not_enter_workflow() {
    let request = request_with_metadata(json!({
        "harness": {
            "service_scene_launch": {
                "scene": "article"
            }
        }
    }));
    let scope = RuntimeSessionScope {
        session_id: "session-1".to_string(),
        thread_id: "thread-1".to_string(),
        turn_id: "turn-1".to_string(),
        workspace_id: None,
    };

    let parsed = parse_image_command_intent(&request, &scope).expect("parse");

    assert!(parsed.is_none());
}

fn request_with_metadata(metadata: Value) -> ExecutionRequest {
    ExecutionRequest {
        host: crate::RuntimeHostContext::default(),
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
        input: agent_runtime::reply_input::RuntimeReplyInput::text("画一张广州夏天的图"),
        runtime_options: Some(RuntimeOptions {
            stream: true,
            runtime_request: Some(app_server_protocol::RuntimeRequest {
                metadata: Some(metadata),
                ..app_server_protocol::RuntimeRequest::default()
            }),
            ..RuntimeOptions::default()
        }),
        expected_output: None,
        structured_output: None,
        output_schema: None,
        event_name: None,
        queued_turn_id: None,
        queue_if_busy: false,
        skip_pre_submit_resume: false,
        agent_control_gateway: None,
        rollout_budget_reminder_source: None,
    }
}
