use super::definition::{
    WorkflowDefinition, WorkflowSourceKind, WorkflowSourceRef, WorkflowStepDefinition,
    WorkflowStepKind, WORKFLOW_SCHEMA_VERSION,
};
use super::read_model::workflow_read_model_from_events;
use super::source_map::{
    workflow_definition_from_manifest, workflow_definition_from_source,
    workflow_step_definitions_from_value, WorkflowDefinitionSourceInput,
};
use super::status::{normalize_workflow_status, WorkflowStatus};
use app_server_protocol::AgentEvent;
use serde_json::json;

#[test]
fn normalizes_legacy_status_values_to_current_contract() {
    assert_eq!(
        normalize_workflow_status("active"),
        Some(WorkflowStatus::Running)
    );
    assert_eq!(
        normalize_workflow_status("pending"),
        Some(WorkflowStatus::Queued)
    );
    assert_eq!(
        normalize_workflow_status("succeeded"),
        Some(WorkflowStatus::Completed)
    );
    assert_eq!(
        normalize_workflow_status("cancelled"),
        Some(WorkflowStatus::Canceled)
    );
    assert_eq!(
        normalize_workflow_status("timeout"),
        Some(WorkflowStatus::Failed)
    );
}

#[test]
fn definition_keeps_source_and_execution_contract_separate() {
    let mut definition = WorkflowDefinition::new(
        "plugin:content:article",
        WorkflowSourceRef {
            source_kind: WorkflowSourceKind::PluginManifest,
            source_id: "content-factory".to_string(),
            source_version: Some("1.0.0".to_string()),
        },
        "content_article_workflow",
        "内容生产",
    );
    definition.task_kind = Some("content.article.generate".to_string());
    definition.steps.push(WorkflowStepDefinition {
        id: "research".to_string(),
        title: "资料检索".to_string(),
        kind: WorkflowStepKind::Connector,
        depends_on: Vec::new(),
        skill_refs: vec!["article-research".to_string()],
        subagent_ref: Some("researcher".to_string()),
        tool_refs: vec!["web.search".to_string()],
        expected_output: Some("写作依据".to_string()),
        input_mapping: None,
        output_mapping: None,
        retry_policy: None,
        human_review: false,
    });

    assert_eq!(definition.schema_version, WORKFLOW_SCHEMA_VERSION);
    assert_eq!(definition.steps[0].kind, WorkflowStepKind::Connector);
    assert_eq!(
        definition.source.source_kind,
        WorkflowSourceKind::PluginManifest
    );
}

#[test]
fn maps_plugin_manifest_steps_to_current_definition() {
    let manifest = json!({
        "id": "content-factory",
        "version": "1.0.0",
        "agentRuntime": {
            "workflows": [
                {
                    "key": "content_article_workflow",
                    "title": "内容生产",
                    "taskKind": "content.article.generate",
                    "outputArtifactKind": "workspace_patch",
                    "steps": [
                        {
                            "id": "research",
                            "title": "资料检索",
                            "subagent": "researcher",
                            "skillRefs": ["article-research"],
                            "connectorRefs": ["web-research"],
                            "expectedOutput": "写作依据"
                        },
                        {
                            "id": "draft",
                            "title": "正文写作",
                            "dependsOn": ["research"],
                            "subagent": "article-writer",
                            "skillRefs": ["article-writing"],
                            "expectedOutput": "articleDraft"
                        }
                    ]
                }
            ]
        }
    });

    let definition = workflow_definition_from_manifest(
        &manifest,
        WorkflowSourceRef {
            source_kind: WorkflowSourceKind::PluginManifest,
            source_id: "content-factory".to_string(),
            source_version: Some("1.0.0".to_string()),
        },
        "content_article_workflow",
    )
    .expect("plugin workflow definition");

    assert_eq!(
        definition.definition_id,
        "plugin_manifest:content-factory:content_article_workflow"
    );
    assert_eq!(definition.workflow_key, "content_article_workflow");
    assert_eq!(
        definition.output_artifact_kind.as_deref(),
        Some("workspace_patch")
    );
    assert_eq!(definition.steps.len(), 2);
    assert_eq!(definition.steps[0].id, "research");
    assert_eq!(definition.steps[0].kind, WorkflowStepKind::Subagent);
    assert_eq!(
        definition.steps[0].subagent_ref.as_deref(),
        Some("researcher")
    );
    assert_eq!(definition.steps[0].skill_refs, vec!["article-research"]);
    assert_eq!(definition.steps[0].tool_refs, vec!["web-research"]);
    assert_eq!(definition.steps[1].depends_on, vec!["research"]);
}

#[test]
fn maps_skill_workflow_steps_as_definition_summary_only() {
    let steps = json!([
        {
            "id": "outline",
            "name": "生成大纲",
            "dependencies": []
        },
        {
            "id": "draft",
            "name": "生成正文",
            "dependencies": ["outline"]
        }
    ]);

    let definition = workflow_definition_from_source(WorkflowDefinitionSourceInput {
        source: WorkflowSourceRef {
            source_kind: WorkflowSourceKind::Skill,
            source_id: "article-writer".to_string(),
            source_version: None,
        },
        workflow_key: "article-writer",
        title: Some("Article Writer"),
        task_kind: None,
        input_schema: None,
        output_artifact_kind: None,
        steps: Some(&steps),
        policies: None,
    })
    .expect("skill workflow summary");

    assert_eq!(
        definition.definition_id,
        "skill:article-writer:article-writer"
    );
    assert_eq!(definition.steps[0].kind, WorkflowStepKind::Skill);
    assert_eq!(definition.steps[1].depends_on, vec!["outline"]);
    assert_eq!(
        definition.policies.as_ref().and_then(|value| value
            .get("summaryOnly")
            .and_then(serde_json::Value::as_bool)),
        Some(true),
        "skill workflow_steps are summary metadata, not executable runtime"
    );
}

#[test]
fn maps_image_command_builtin_workflow_to_same_definition_schema() {
    let steps = json!([
        {
            "id": "intent",
            "title": "解析图片需求",
            "kind": "agent_task"
        },
        {
            "id": "route",
            "title": "确认图片模型",
            "depends_on": ["intent"],
            "toolRefs": ["image.route"]
        },
        {
            "id": "create_tasks",
            "title": "创建图片任务",
            "dependsOn": ["route"],
            "kind": "tool",
            "toolRefs": ["mediaTaskArtifact/imageCreate"]
        },
        {
            "id": "generate",
            "title": "生成图片",
            "dependsOn": ["create_tasks"],
            "kind": "connector"
        },
        {
            "id": "persist_outputs",
            "title": "保存结果",
            "dependsOn": ["generate"],
            "kind": "storage"
        }
    ]);

    let definition = workflow_definition_from_source(WorkflowDefinitionSourceInput {
        source: WorkflowSourceRef {
            source_kind: WorkflowSourceKind::ImageCommand,
            source_id: "image-command".to_string(),
            source_version: None,
        },
        workflow_key: "image_command_workflow",
        title: Some("图片生成"),
        task_kind: Some("image.generate"),
        input_schema: None,
        output_artifact_kind: Some("image_task"),
        steps: Some(&steps),
        policies: None,
    })
    .expect("image workflow definition");

    assert_eq!(
        definition.definition_id,
        "image_command:image-command:image_command_workflow"
    );
    assert_eq!(
        definition
            .steps
            .iter()
            .map(|step| step.id.as_str())
            .collect::<Vec<_>>(),
        vec![
            "intent",
            "route",
            "create_tasks",
            "generate",
            "persist_outputs"
        ]
    );
    assert_eq!(definition.steps[1].kind, WorkflowStepKind::Tool);
    assert_eq!(definition.steps[4].kind, WorkflowStepKind::Storage);
}

#[test]
fn maps_content_factory_orchestration_as_definition_source() {
    let orchestration = json!([
        {
            "id": "research",
            "title": "资料检索",
            "subagent": "researcher",
            "skillRefs": ["article-research"],
            "connectorRefs": ["lime-knowledge", "web-research"],
            "expectedOutput": "sources"
        },
        {
            "id": "draft",
            "title": "文章生成",
            "dependsOn": ["research"],
            "subagent": "article-writer",
            "skillRefs": ["article-writing"],
            "expectedOutput": "articleDraft"
        },
        {
            "id": "review",
            "title": "人工复核",
            "dependsOn": ["draft"],
            "kind": "human_review",
            "humanReview": true
        }
    ]);

    let definition = workflow_definition_from_source(WorkflowDefinitionSourceInput {
        source: WorkflowSourceRef {
            source_kind: WorkflowSourceKind::ContentFactory,
            source_id: "content-factory-worker".to_string(),
            source_version: None,
        },
        workflow_key: "content_article_workflow",
        title: Some("内容工厂文章工作流"),
        task_kind: Some("content.article.generate"),
        input_schema: None,
        output_artifact_kind: Some("workspace_patch"),
        steps: Some(&orchestration),
        policies: Some(&json!({
            "hookPolicy": {
                "task.complete": ["persist-worker-evidence"]
            }
        })),
    })
    .expect("content factory workflow definition");

    assert_eq!(
        definition.source.source_kind,
        WorkflowSourceKind::ContentFactory
    );
    assert_eq!(
        definition.steps[0].tool_refs,
        vec!["lime-knowledge", "web-research"]
    );
    assert_eq!(definition.steps[2].kind, WorkflowStepKind::ManualGate);
    assert!(definition.steps[2].human_review);
    assert_eq!(
        definition
            .policies
            .as_ref()
            .and_then(|value| value.pointer("/hookPolicy/task.complete/0"))
            .and_then(serde_json::Value::as_str),
        Some("persist-worker-evidence")
    );
}

#[test]
fn maps_test_only_workflow_runtime_host_dsl_kinds_to_current_step_kinds() {
    let dsl_steps = json!([
        {
            "id": "save-input",
            "label": "保存输入",
            "kind": "storage.set"
        },
        {
            "id": "lookup",
            "label": "检索资料",
            "kind": "knowledge.search"
        },
        {
            "id": "task",
            "label": "启动任务",
            "kind": "agent.startTask",
            "humanReview": true
        },
        {
            "id": "artifact",
            "label": "生成产物",
            "kind": "artifacts.create"
        },
        {
            "id": "evidence",
            "label": "记录证据",
            "kind": "evidence.record"
        }
    ]);

    let steps = workflow_step_definitions_from_value(Some(&dsl_steps), None);

    assert_eq!(
        steps.iter().map(|step| &step.kind).collect::<Vec<_>>(),
        vec![
            &WorkflowStepKind::Storage,
            &WorkflowStepKind::Connector,
            &WorkflowStepKind::AgentTask,
            &WorkflowStepKind::Artifact,
            &WorkflowStepKind::Evidence,
        ],
        "WorkflowRuntimeHost DSL is adapter/test-only input, not a current runtime standard"
    );
    assert!(steps[2].human_review);
}

#[test]
fn projects_workflow_run_and_steps_from_runtime_events() {
    let events = vec![
        event(
            1,
            "workflow.run.started",
            json!({
                "workflowRunId": "task-article:workflow",
                "workflowKey": "content_article_workflow",
                "workflowTitle": "内容生产",
                "taskId": "task-article",
                "turnId": "turn-1",
                "appId": "content-factory",
                "steps": [
                    {
                        "id": "research",
                        "title": "资料检索",
                        "status": "pending",
                        "index": 0,
                        "stepCount": 2
                    },
                    {
                        "id": "draft",
                        "title": "正文写作",
                        "status": "pending",
                        "index": 1,
                        "stepCount": 2
                    }
                ]
            }),
        ),
        event(
            2,
            "workflow.step.started",
            json!({
                "workflowRunId": "task-article:workflow",
                "workflowKey": "content_article_workflow",
                "stepId": "research",
                "stepTitle": "资料检索",
                "stepIndex": 0,
                "stepCount": 2,
                "status": "running"
            }),
        ),
        event(
            3,
            "workflow.step.completed",
            json!({
                "workflowRunId": "task-article:workflow",
                "workflowKey": "content_article_workflow",
                "stepId": "research",
                "stepTitle": "资料检索",
                "artifactRefs": ["artifact-research"],
                "evidenceRefs": ["evidence-research"],
                "status": "accepted"
            }),
        ),
        event(
            4,
            "workflow.run.completed",
            json!({
                "workflowRunId": "task-article:workflow",
                "workflowKey": "content_article_workflow",
                "status": "task_created"
            }),
        ),
    ];

    let read_model = workflow_read_model_from_events(&events);

    assert_eq!(
        read_model.active_workflow_run_id.as_deref(),
        None,
        "terminal run should clear active workflow"
    );
    assert_eq!(read_model.workflow_runs.len(), 1);
    let run = &read_model.workflow_runs[0];
    assert_eq!(run.workflow_run_id, "task-article:workflow");
    assert_eq!(run.status, WorkflowStatus::Completed);
    assert_eq!(run.step_counts.total, 2);
    assert_eq!(run.step_counts.completed, 1);
    assert_eq!(run.step_counts.queued, 1);

    let research = read_model
        .workflow_steps
        .iter()
        .find(|step| step.step_id == "research")
        .expect("research step");
    assert_eq!(research.status, WorkflowStatus::Completed);
    assert_eq!(research.artifact_refs, vec!["artifact-research"]);
    assert_eq!(research.evidence_refs, vec!["evidence-research"]);

    let draft = read_model
        .workflow_steps
        .iter()
        .find(|step| step.step_id == "draft")
        .expect("draft step");
    assert_eq!(draft.status, WorkflowStatus::Queued);
}

#[test]
fn projects_retry_actions_and_retry_attempts_from_runtime_events() {
    let events = vec![
        event(
            1,
            "workflow.run.started",
            json!({
                "workflowRunId": "task-article:workflow",
                "workflowKey": "content_article_workflow",
                "workflowTitle": "内容生产",
                "status": "running"
            }),
        ),
        event(
            2,
            "workflow.step.failed",
            json!({
                "workflowRunId": "task-article:workflow",
                "stepId": "draft",
                "stepTitle": "正文写作",
                "attempt": 1,
                "status": "failed",
                "failure": {
                    "reason": "model_error"
                }
            }),
        ),
    ];

    let read_model = workflow_read_model_from_events(&events);

    assert!(read_model.actions.iter().any(|action| {
        action.action_type == "retry"
            && action.workflow_run_id == "task-article:workflow"
            && action.step_id.as_deref() == Some("draft")
    }));

    let retry_events = vec![
        events[0].clone(),
        events[1].clone(),
        event(
            4,
            "workflow.step.retrying",
            json!({
                "workflowRunId": "task-article:workflow",
                "stepId": "draft",
                "stepTitle": "正文写作",
                "attempt": 2,
                "status": "retrying"
            }),
        ),
        event(
            5,
            "workflow.run.retrying",
            json!({
                "workflowRunId": "task-article:workflow",
                "workflowKey": "content_article_workflow",
                "workflowTitle": "内容生产",
                "status": "retrying"
            }),
        ),
    ];

    let retrying = workflow_read_model_from_events(&retry_events);
    let step = retrying
        .workflow_steps
        .iter()
        .find(|step| step.step_id == "draft")
        .expect("draft step");
    assert_eq!(step.status, WorkflowStatus::Retrying);
    assert_eq!(step.attempt, 2);
    assert!(step.failure.is_none());
    assert!(step.finished_at.is_none());
    assert_eq!(retrying.workflow_runs[0].status, WorkflowStatus::Retrying);
    assert!(retrying.workflow_runs[0].failure.is_none());
    assert!(retrying.workflow_runs[0].finished_at.is_none());
}

#[test]
fn projects_resume_events_as_running_workflow_state() {
    let events = vec![
        event(
            1,
            "workflow.step.resuming",
            json!({
                "workflowRunId": "task-article:workflow",
                "workflowKey": "content_article_workflow",
                "stepId": "draft",
                "stepTitle": "正文写作",
                "status": "resuming"
            }),
        ),
        event(
            2,
            "workflow.run.resuming",
            json!({
                "workflowRunId": "task-article:workflow",
                "workflowKey": "content_article_workflow",
                "workflowTitle": "内容生产",
                "status": "resuming"
            }),
        ),
    ];

    let read_model = workflow_read_model_from_events(&events);

    assert_eq!(
        read_model.active_workflow_run_id.as_deref(),
        Some("task-article:workflow")
    );
    assert_eq!(read_model.workflow_runs[0].status, WorkflowStatus::Running);
    assert!(read_model.workflow_runs[0].finished_at.is_none());
    assert_eq!(read_model.workflow_steps[0].status, WorkflowStatus::Running);
    assert!(read_model.workflow_steps[0].finished_at.is_none());
}

fn event(sequence: u64, event_type: &str, payload: serde_json::Value) -> AgentEvent {
    AgentEvent {
        event_id: format!("event-{sequence}"),
        sequence,
        session_id: "session-1".to_string(),
        thread_id: Some("thread-1".to_string()),
        turn_id: Some("turn-1".to_string()),
        event_type: event_type.to_string(),
        timestamp: format!("2026-07-04T00:00:0{sequence}.000Z"),
        payload,
    }
}
