use super::*;
use agent_protocol::{
    ApprovalAction, ApprovalDecision, ApprovalScope, CollabAgentOperation, FileChange,
    FileChangeKind, FileChangeStatus, ItemId, ItemStatus, MessageContentPart,
    MessageContentReference, SessionId, SubAgentActivityKind, Thread, ThreadHistoryChangeSet,
    ThreadId, ThreadItem, ThreadItemPayload, ThreadStatus, ThreadTurnsView, Turn,
    TurnAdmissionState, TurnApprovalState, TurnId, TurnItemsView, TurnQueueState, TurnStatus,
};
use app_server_protocol::{AgentEvent, AgentSession, AgentTurn, AgentTurnStatus};
use futures::executor::block_on;
use serde_json::json;
use thread_store::{ApplyThreadHistoryParams, CreateThreadParams, ThreadStore};

fn stored_running_session(started_at: &str, latest_event_at: &str) -> StoredSession {
    let session_id = "sess_read_model_orphan_running".to_string();
    let thread_id = "thread_read_model_orphan_running".to_string();
    let turn_id = "turn_read_model_orphan_running".to_string();
    StoredSession {
        session: AgentSession {
            session_id: session_id.clone(),
            thread_id: thread_id.clone(),
            app_id: "agent-chat".to_string(),
            workspace_id: Some("workspace-current".to_string()),
            business_object_ref: None,
            status: app_server_protocol::AgentSessionStatus::Running,
            created_at: started_at.to_string(),
            updated_at: latest_event_at.to_string(),
        },
        turns: vec![AgentTurn {
            turn_id: turn_id.clone(),
            session_id: session_id.clone(),
            thread_id: thread_id.clone(),
            status: AgentTurnStatus::Running,
            started_at: Some(started_at.to_string()),
            completed_at: None,
        }],
        turn_inputs: std::collections::HashMap::new(),
        turn_runtime_options: std::collections::HashMap::new(),
        events: vec![AgentEvent {
            event_id: "event-read-model-running".to_string(),
            sequence: 1,
            session_id,
            thread_id: Some(thread_id),
            turn_id: Some(turn_id),
            event_type: "turn.started".to_string(),
            timestamp: latest_event_at.to_string(),
            payload: json!({}),
        }],
        output_blobs: std::collections::HashMap::new(),
    }
}

#[test]
fn canonical_read_keeps_runtime_warning_message_and_localization_code() {
    let mut stored = stored_running_session("2026-07-31T00:00:00Z", "2026-07-31T00:00:01Z");
    stored.events.push(AgentEvent {
        event_id: "event-read-model-warning".to_string(),
        sequence: 2,
        session_id: stored.session.session_id.clone(),
        thread_id: Some(stored.session.thread_id.clone()),
        turn_id: Some(stored.turns[0].turn_id.clone()),
        event_type: "runtime.warning".to_string(),
        timestamp: "2026-07-31T00:00:02Z".to_string(),
        payload: json!({
            "code": "skill_not_available",
            "message": "技能不可用，已继续执行。"
        }),
    });

    let detail = runtime_session_read_detail_with_item_source(
        &stored,
        ReadDetailOptions::default(),
        &[],
        Some(&[]),
    );
    let warning = detail["items"]
        .as_array()
        .and_then(|items| items.iter().find(|item| item["type"] == "warning"))
        .expect("warning item");

    assert_eq!(warning["thread_id"], stored.session.thread_id);
    assert_eq!(warning["turn_id"], stored.turns[0].turn_id);
    assert_eq!(warning["message"], "技能不可用，已继续执行。");
    assert_eq!(warning["code"], "skill_not_available");
    assert_eq!(detail["thread_read"]["thread_items"], detail["items"]);
}

#[test]
fn canonical_read_rejects_malformed_runtime_warning_shapes_and_identity() {
    let mut stored = stored_running_session("2026-07-31T00:00:00Z", "2026-07-31T00:00:01Z");
    let session_id = stored.session.session_id.clone();
    let thread_id = stored.session.thread_id.clone();
    let turn_id = stored.turns[0].turn_id.clone();
    let warning = |event_id: &str,
                   thread_id: Option<String>,
                   turn_id: Option<String>,
                   payload: serde_json::Value| AgentEvent {
        event_id: event_id.to_string(),
        sequence: 2,
        session_id: session_id.clone(),
        thread_id,
        turn_id,
        event_type: "runtime.warning".to_string(),
        timestamp: "2026-07-31T00:00:02Z".to_string(),
        payload,
    };
    stored.events.extend([
        warning(
            "event-warning-alias-message",
            Some(thread_id.clone()),
            Some(turn_id.clone()),
            json!({ "warning": "legacy alias must not recover" }),
        ),
        warning(
            "event-warning-invalid-code",
            Some(thread_id.clone()),
            Some(turn_id.clone()),
            json!({ "message": "invalid code", "code": 42 }),
        ),
        warning(
            "event-warning-missing-thread",
            None,
            Some(turn_id.clone()),
            json!({ "message": "missing thread identity" }),
        ),
        warning(
            "event-warning-missing-turn",
            Some(thread_id),
            None,
            json!({ "message": "missing turn identity" }),
        ),
    ]);

    let detail = runtime_session_read_detail_with_item_source(
        &stored,
        ReadDetailOptions::default(),
        &[],
        Some(&[]),
    );

    assert!(detail["items"]
        .as_array()
        .is_some_and(|items| items.iter().all(|item| item["type"] != "warning")));
}

#[test]
fn workflow_respond_action_requires_matching_canonical_pending_action() {
    let session_id = "sess_workflow_action_read".to_string();
    let thread_id = "thread_workflow_action_read".to_string();
    let turn_id = "turn_workflow_action_read".to_string();
    let workflow_events = vec![AgentEvent {
        event_id: "event-workflow-run".to_string(),
        sequence: 1,
        session_id: session_id.clone(),
        thread_id: Some(thread_id.clone()),
        turn_id: None,
        event_type: "workflow.run.started".to_string(),
        timestamp: "2026-07-15T05:00:00Z".to_string(),
        payload: json!({
            "workflowRunId": "run-review",
            "workflowKey": "content_article_workflow",
            "turnId": turn_id,
            "status": "running",
            "steps": [{
                "stepId": "review",
                "stepTitle": "Review",
                "status": "waiting",
                "requestId": "ask-review",
                "actionType": "ask_user"
            }]
        }),
    }];
    let mut stored = StoredSession {
        session: AgentSession {
            session_id: session_id.clone(),
            thread_id: thread_id.clone(),
            app_id: "content-factory-app".to_string(),
            workspace_id: None,
            business_object_ref: None,
            status: app_server_protocol::AgentSessionStatus::Running,
            created_at: "2026-07-15T05:00:00Z".to_string(),
            updated_at: "2026-07-15T05:00:00Z".to_string(),
        },
        turns: Vec::new(),
        turn_inputs: std::collections::HashMap::new(),
        turn_runtime_options: std::collections::HashMap::new(),
        events: workflow_events.clone(),
        output_blobs: std::collections::HashMap::new(),
    };

    let without_pending = workflow_read_model_from_stored_session(&stored, &[]);
    assert!(
        without_pending
            .actions
            .iter()
            .all(|action| action.action_type != "respond"),
        "workflow step metadata must not manufacture a respond action"
    );

    stored.session.status = app_server_protocol::AgentSessionStatus::WaitingAction;
    stored.turns.push(AgentTurn {
        turn_id: turn_id.clone(),
        session_id: session_id.clone(),
        thread_id: thread_id.clone(),
        status: AgentTurnStatus::WaitingAction,
        started_at: Some("2026-07-15T05:00:00Z".to_string()),
        completed_at: None,
    });
    let mut tool_confirmation_workflow_events = workflow_events.clone();
    tool_confirmation_workflow_events[0].payload["steps"][0]["actionType"] =
        json!("tool_confirmation");
    stored.events = vec![AgentEvent {
        event_id: "event-action-required-invalid-confirmation".to_string(),
        sequence: 1,
        session_id: session_id.clone(),
        thread_id: Some(thread_id.clone()),
        turn_id: Some(turn_id.clone()),
        event_type: "action.required".to_string(),
        timestamp: "2026-07-15T05:00:00Z".to_string(),
        payload: json!({
            "requestId": "ask-review",
            "actionType": "tool_confirmation",
            "toolCallId": "tool-review",
            "prompt": "Review the draft",
            "deadlineAtMs": 4_102_444_800_000_u64
        }),
    }];
    stored.events.extend(tool_confirmation_workflow_events);

    let without_restorable_confirmation = workflow_read_model_from_stored_session(&stored, &[]);
    assert!(
        without_restorable_confirmation
            .actions
            .iter()
            .all(|action| action.action_type != "respond"),
        "workflow/read must not publish a tool confirmation that cold restore rejects"
    );

    stored.events = vec![AgentEvent {
        event_id: "event-action-required".to_string(),
        sequence: 1,
        session_id,
        thread_id: Some(thread_id),
        turn_id: Some(turn_id),
        event_type: "action.required".to_string(),
        timestamp: "2026-07-15T05:00:00Z".to_string(),
        payload: json!({
            "requestId": "ask-review",
            "actionType": "ask_user",
            "prompt": "Review the draft",
            "deadlineAtMs": 4_102_444_800_000_u64
        }),
    }];
    stored.events.extend(workflow_events);

    let with_pending = workflow_read_model_from_stored_session(&stored, &[]);
    assert!(with_pending.actions.iter().any(|action| {
        action.action_type == "respond"
            && action.workflow_run_id == "run-review"
            && action.step_id.as_deref() == Some("review")
            && action.request_id.as_deref() == Some("ask-review")
            && action.agent_action_type.as_deref() == Some("ask_user")
    }));
}

#[test]
fn thread_read_downgrades_stale_orphan_running_turn() {
    let stored = stored_running_session("2026-03-29T00:00:00.000Z", "2026-03-29T00:00:01.000Z");

    let thread_read =
        runtime_thread_read_from_stored_session_with_usage_events(&stored, None, Vec::new(), &[]);

    assert_eq!(thread_read["status"], "idle");
    assert_eq!(thread_read["active_turn_id"], serde_json::Value::Null);
    assert_eq!(thread_read["diagnostics"]["latest_turn_status"], "running");
}

#[test]
fn thread_read_keeps_recent_running_turn_active() {
    let now = chrono::Utc::now().to_rfc3339();
    let stored = stored_running_session(now.as_str(), now.as_str());

    let thread_read =
        runtime_thread_read_from_stored_session_with_usage_events(&stored, None, Vec::new(), &[]);

    assert_eq!(thread_read["status"], "running");
    assert_eq!(
        thread_read["active_turn_id"],
        "turn_read_model_orphan_running"
    );
}

#[test]
fn canonical_approval_detail_uses_typed_terminal_response() {
    let approval_item = |decision: Option<ApprovalDecision>| ThreadItem {
        session_id: SessionId::new("session-approval-read"),
        thread_id: ThreadId::new("thread-approval-read"),
        turn_id: TurnId::new("turn-approval-read"),
        item_id: ItemId::new("approval-read"),
        sequence: 3,
        ordinal: 2,
        created_at_ms: 1,
        updated_at_ms: 2,
        completed_at_ms: decision.map(|_| 2),
        kind: agent_protocol::ItemKind::Approval,
        status: if decision.is_some() {
            ItemStatus::Completed
        } else {
            ItemStatus::InProgress
        },
        payload: ThreadItemPayload::Approval {
            request_id: "approval-read".to_string(),
            action: ApprovalAction {
                kind: "tool_confirmation".to_string(),
                description: "Allow command?".to_string(),
            },
            scope: ApprovalScope::Session,
            available_decisions: vec![ApprovalDecision::ApprovedForSession],
            decision,
            requested_at_ms: Some(1),
            resolved_at_ms: decision.map(|_| 2),
            reason_code: Some("user_decision".to_string()),
            expires_at_ms: None,
        },
        metadata: json!({}),
    };

    let terminal =
        canonical_item_to_agent_detail(&approval_item(Some(ApprovalDecision::ApprovedForSession)));
    assert_eq!(
        terminal["response"],
        json!({
            "decision": "approvedForSession",
            "decision_scope": "session",
            "reason_code": "user_decision",
        })
    );
    assert!(terminal.get("reason_code").is_none());

    let pending = canonical_item_to_agent_detail(&approval_item(None));
    assert!(pending.get("response").is_none());
}

#[test]
fn canonical_wait_collab_tool_projects_as_completed_tool_call() {
    let item = ThreadItem {
        session_id: SessionId::new("session-wait-read"),
        thread_id: ThreadId::new("thread-wait-read"),
        turn_id: TurnId::new("turn-wait-read"),
        item_id: ItemId::new("wait-read"),
        sequence: 1,
        ordinal: 0,
        created_at_ms: 1,
        updated_at_ms: 2,
        completed_at_ms: Some(2),
        kind: agent_protocol::ItemKind::CollabAgentToolCall,
        status: ItemStatus::Completed,
        payload: ThreadItemPayload::CollabAgentToolCall {
            call_id: "wait-read".to_string(),
            operation: CollabAgentOperation::Wait,
            target_thread_id: None,
            message: None,
            output: None,
            agent_states: Default::default(),
        },
        metadata: json!({}),
    };

    let detail = canonical_item_to_agent_detail(&item);

    assert_eq!(detail["type"], "tool_call");
    assert_eq!(detail["status"], "completed");
    assert_eq!(detail["call_id"], "wait-read");
    assert_eq!(detail["tool_name"], "wait_agent");
    assert_eq!(detail["arguments"], json!([]));
    assert!(detail.get("status_label").is_none());
}

#[test]
fn canonical_file_change_matches_v2_patch_projection_for_decline() {
    let item = ThreadItem {
        session_id: SessionId::new("session-patch-read"),
        thread_id: ThreadId::new("thread-patch-read"),
        turn_id: TurnId::new("turn-patch-read"),
        item_id: ItemId::new("patch-read"),
        sequence: 1,
        ordinal: 0,
        created_at_ms: 1,
        updated_at_ms: 2,
        completed_at_ms: Some(2),
        kind: agent_protocol::ItemKind::File,
        status: ItemStatus::Completed,
        payload: ThreadItemPayload::File {
            changes: vec![
                FileChange {
                    path: "source.txt".to_string(),
                    kind: FileChangeKind::Update {
                        move_path: Some("target.txt".to_string()),
                    },
                    diff: "-before\n+after".to_string(),
                },
                FileChange {
                    path: "dead.txt".to_string(),
                    kind: FileChangeKind::Delete,
                    diff: "-dead".to_string(),
                },
            ],
            status: FileChangeStatus::Rejected,
        },
        metadata: json!({}),
    };

    let detail = canonical_item_to_agent_detail(&item);

    assert_eq!(detail["type"], "patch");
    assert_eq!(detail["status"], "failed");
    assert_eq!(detail["success"], false);
    assert_eq!(detail["paths"], json!(["source.txt", "dead.txt"]));
    assert_eq!(detail["changes"][0]["path"], "source.txt");
    assert_eq!(detail["changes"][0]["kind"]["type"], "update");
    assert_eq!(detail["changes"][0]["kind"]["move_path"], "target.txt");
}

#[test]
fn read_detail_projects_thread_items_into_thread_read() {
    let session_id = "sess_read_model_thread_items".to_string();
    let thread_id = "thread_read_model_thread_items".to_string();
    let turn_id = "turn_read_model_thread_items".to_string();
    let stored = StoredSession {
        session: AgentSession {
            session_id: session_id.clone(),
            thread_id: thread_id.clone(),
            app_id: "agent-chat".to_string(),
            workspace_id: Some("workspace-current".to_string()),
            business_object_ref: None,
            status: app_server_protocol::AgentSessionStatus::Completed,
            created_at: "2026-03-29T00:00:00.000Z".to_string(),
            updated_at: "2026-03-29T00:00:02.000Z".to_string(),
        },
        turns: vec![AgentTurn {
            turn_id: turn_id.clone(),
            session_id: session_id.clone(),
            thread_id: thread_id.clone(),
            status: AgentTurnStatus::Completed,
            started_at: Some("2026-03-29T00:00:00.000Z".to_string()),
            completed_at: Some("2026-03-29T00:00:02.000Z".to_string()),
        }],
        turn_inputs: std::collections::HashMap::new(),
        turn_runtime_options: std::collections::HashMap::new(),
        events: vec![
            AgentEvent {
                event_id: "evt-read-model-user-message".to_string(),
                sequence: 0,
                session_id: session_id.clone(),
                thread_id: Some(thread_id.clone()),
                turn_id: Some(turn_id.clone()),
                event_type: "message.created".to_string(),
                timestamp: "2026-03-29T00:00:00.500Z".to_string(),
                payload: json!({
                    "role": "user",
                    "visibility": "user_visible",
                    "input": [{"type": "text", "text": "恢复历史用户输入"}],
                    "content": {
                        "kind": "inline_text",
                        "text": "恢复历史用户输入"
                    },
                    "textElements": [
                        {
                            "type": "text",
                            "text": "保留富文本输入片段"
                        }
                    ],
                    "text_elements": [
                        {
                            "type": "text",
                            "text": "保留富文本输入片段"
                        }
                    ]
                }),
            },
            AgentEvent {
                event_id: "evt-read-model-reasoning-item".to_string(),
                sequence: 1,
                session_id,
                thread_id: Some(thread_id),
                turn_id: Some(turn_id),
                event_type: "item.started".to_string(),
                timestamp: "2026-03-29T00:00:01.000Z".to_string(),
                payload: json!({
                    "item": {
                        "id": "reasoning-read-model-thread-items",
                        "thread_id": "thread_read_model_thread_items",
                        "turn_id": "turn_read_model_thread_items",
                        "sequence": 1,
                        "status": "completed",
                        "type": "reasoning",
                        "text": "先恢复历史推理项",
                        "summary": ["先恢复历史推理项"]
                    }
                }),
            },
        ],
        output_blobs: std::collections::HashMap::new(),
    };

    let detail =
        runtime_session_read_detail_with_options(&stored, ReadDetailOptions::default(), &[]);

    assert_eq!(detail["items"], detail["thread_read"]["thread_items"]);
    assert_eq!(
        detail["thread_read"]["thread_items"][0]["id"],
        "reasoning-read-model-thread-items"
    );
    assert_eq!(
        detail["messages"][0]["textElements"][0]["text"],
        "保留富文本输入片段"
    );
    assert_eq!(
        detail["messages"][0]["text_elements"][0]["text"],
        "保留富文本输入片段"
    );
    assert!(detail["messages"][0]["content"]
        .as_array()
        .expect("message content")
        .iter()
        .any(|part| part["text"] == "保留富文本输入片段"));
}

#[test]
fn canonical_overlay_preserves_richer_current_coding_projection() {
    const FILE_PATH: &str = ".lime/qc/code-artifact-workbench-host-fixture/src/coding-target.ts";
    const COMMAND_ID: &str = "code-artifact-workbench-host:command:test";
    const TEST_RUN_ID: &str = "code-artifact-workbench-host:test:unit";
    const FAILURE_PREVIEW: &str =
        "FAIL coding-target.test.ts: expected codingWorkbenchSmoke to be true";

    let mut stored = stored_running_session("2026-07-15T22:54:38.000Z", "2026-07-15T22:54:39.000Z");
    let session_id = stored.session.session_id.clone();
    let thread_id = stored.session.thread_id.clone();
    let turn_id = stored.turns[0].turn_id.clone();
    let event = |sequence: u64, event_type: &str, payload: serde_json::Value| AgentEvent {
        event_id: format!("event-coding-{sequence}"),
        sequence,
        session_id: session_id.clone(),
        thread_id: Some(thread_id.clone()),
        turn_id: Some(turn_id.clone()),
        event_type: event_type.to_string(),
        timestamp: format!("2026-07-15T22:54:{sequence:02}.000Z"),
        payload,
    };
    stored.events = vec![
        event(1, "file.changed", json!({ "path": FILE_PATH })),
        event(
            2,
            "command.started",
            json!({ "commandId": COMMAND_ID, "command": "npm test -- coding-target", "cwd": "." }),
        ),
        event(
            3,
            "command.output",
            json!({ "commandId": COMMAND_ID, "preview": FAILURE_PREVIEW }),
        ),
        event(
            4,
            "command.exited",
            json!({
                "commandId": COMMAND_ID,
                "command": "npm test -- coding-target",
                "exitCode": 1,
                "preview": FAILURE_PREVIEW
            }),
        ),
        event(
            5,
            "test.started",
            json!({ "testRunId": TEST_RUN_ID, "commandId": COMMAND_ID, "suite": "coding-target" }),
        ),
        event(
            6,
            "test.completed",
            json!({
                "testRunId": TEST_RUN_ID,
                "commandId": COMMAND_ID,
                "suite": "coding-target",
                "result": "failed",
                "passed": 0,
                "failed": 1
            }),
        ),
    ];
    let canonical_items = vec![
        json!({
            "id": format!("item_{COMMAND_ID}"),
            "type": "command_execution",
            "turn_id": turn_id,
            "status": "completed",
            "command": "npm test -- coding-target",
            "cwd": ".",
            "exit_code": 1,
            "aggregated_output": null
        }),
        json!({
            "id": "item_coding-target",
            "type": "patch",
            "turn_id": stored.turns[0].turn_id,
            "status": "completed",
            "changes": [{
                "path": FILE_PATH,
                "kind": { "type": "update", "move_path": null },
                "diff": ""
            }],
            "paths": [FILE_PATH],
            "success": true
        }),
    ];

    let detail = runtime_session_read_detail_with_item_source(
        &stored,
        ReadDetailOptions::default(),
        &[],
        Some(&canonical_items),
    );
    let serialized = serde_json::to_string(&detail).expect("serialize read detail");

    assert!(serialized.contains(FILE_PATH));
    assert!(serialized.contains(COMMAND_ID));
    assert!(serialized.contains(TEST_RUN_ID));
    assert!(serialized.contains(FAILURE_PREVIEW));
    assert_eq!(
        detail["thread_read"]["commands"].as_array().map(Vec::len),
        Some(1)
    );
    assert_eq!(
        detail["thread_read"]["commands"][0]["command_id"],
        COMMAND_ID
    );
    assert_eq!(
        detail["thread_read"]["commands"][0]["output_preview"],
        FAILURE_PREVIEW
    );
}

#[test]
fn command_lifecycle_materializes_complete_canonical_snapshot() {
    const COMMAND_ID: &str = "command-canonical-snapshot";
    const OUTPUT: &str = "FAIL canonical command output";
    let event = |sequence: u64, event_type: &str, payload: serde_json::Value| AgentEvent {
        event_id: format!("event-command-{sequence}"),
        sequence,
        session_id: "session-command-snapshot".to_string(),
        thread_id: Some("thread-command-snapshot".to_string()),
        turn_id: Some("turn-command-snapshot".to_string()),
        event_type: event_type.to_string(),
        timestamp: format!("2026-07-15T23:00:{sequence:02}.000Z"),
        payload,
    };
    let changes = super::super::thread_item_projection::materialize_events(
        &[
            event(
                1,
                "command.started",
                json!({ "commandId": COMMAND_ID, "command": "npm test", "cwd": "." }),
            ),
            event(
                2,
                "command.output",
                json!({ "commandId": COMMAND_ID, "preview": OUTPUT }),
            ),
            event(
                3,
                "command.exited",
                json!({ "commandId": COMMAND_ID, "exitCode": 1 }),
            ),
        ],
        "session-command-snapshot",
        "thread-command-snapshot",
    )
    .expect("materialize command lifecycle");

    assert_eq!(changes.changed_items.len(), 1);
    let item = &changes.changed_items[0];
    assert_eq!(item.metadata["source_call_id"], COMMAND_ID);
    assert_eq!(item.status, ItemStatus::Completed);
    assert_eq!(
        item.payload,
        ThreadItemPayload::Command {
            command: "npm test".to_string(),
            cwd: Some(".".to_string()),
            output: Some(OUTPUT.to_string()),
            exit_code: Some(1),
        }
    );
}

#[test]
fn read_detail_prefers_canonical_thread_store_items_after_restart() {
    let temp = tempfile::tempdir().expect("tempdir");
    let database_path = temp.path().join("projection.sqlite");
    let projection_store = ProjectionStore::initialize(database_path.clone()).expect("store");
    let stored = stored_running_session("2026-03-29T00:00:00.000Z", "2026-03-29T00:00:01.000Z");
    let thread = Thread {
        session_id: SessionId::new(stored.session.session_id.clone()),
        thread_id: ThreadId::new(stored.session.thread_id.clone()),
        status: ThreadStatus::Idle,
        created_at_ms: 1,
        updated_at_ms: 2,
        archived: false,
        recency_at_ms: Some(2),
        parent_thread_id: None,
        agent_path: None,
        agent_nickname: None,
        agent_role: None,
        last_task_message: None,
        agent_state: None,
        forked_from_id: None,
        preview: String::new(),
        model_provider: "test".to_string(),
        product: None,
        name: None,
        metadata: json!({}),
        turns: Vec::new(),
        turns_view: ThreadTurnsView::NotLoaded,
    };
    block_on(projection_store.create_thread(CreateThreadParams {
        thread: thread.clone(),
    }))
    .expect("create thread");
    let turn = Turn {
        session_id: thread.session_id.clone(),
        thread_id: thread.thread_id.clone(),
        turn_id: TurnId::new("turn-read-model-canonical"),
        status: TurnStatus::Completed,
        admission: TurnAdmissionState::Accepted,
        queue: TurnQueueState::Running,
        approval: TurnApprovalState::NotRequired,
        items: Vec::new(),
        items_view: TurnItemsView::NotLoaded,
        error: None,
        created_at_ms: 1,
        updated_at_ms: 2,
        started_at_ms: Some(1),
        completed_at_ms: Some(2),
        duration_ms: Some(1),
    };
    let item = ThreadItem {
        session_id: thread.session_id.clone(),
        thread_id: thread.thread_id.clone(),
        turn_id: turn.turn_id.clone(),
        item_id: ItemId::new("message-read-model-canonical"),
        sequence: 2,
        ordinal: 1,
        created_at_ms: 1,
        updated_at_ms: 2,
        completed_at_ms: Some(2),
        kind: agent_protocol::ItemKind::AgentMessage,
        status: ItemStatus::Completed,
        payload: ThreadItemPayload::AgentMessage {
            text: "canonical item".to_string(),
            phase: None,
            content_parts: vec![
                MessageContentPart::Text {
                    text: "canonical item".to_string(),
                },
                MessageContentPart::Media {
                    kind: "image".to_string(),
                    reference: MessageContentReference {
                        uri: "sidecar://media/read-model".to_string(),
                        mime_type: "image/png".to_string(),
                        title: Some("read model image".to_string()),
                        source_uri: None,
                        source_path: Some("/tmp/media/read-model.png".to_string()),
                        preview_url: None,
                        sidecar_ref: None,
                        sha256: Some("abc123".to_string()),
                        byte_size: Some(4),
                    },
                    caption: Some("result image".to_string()),
                },
            ],
        },
        metadata: json!({}),
    };
    let subagent_item =
        |item_id: &str, sequence: u64, ordinal: u64, activity: SubAgentActivityKind| ThreadItem {
            session_id: thread.session_id.clone(),
            thread_id: thread.thread_id.clone(),
            turn_id: turn.turn_id.clone(),
            item_id: ItemId::new(item_id),
            sequence,
            ordinal,
            created_at_ms: sequence as i64,
            updated_at_ms: sequence as i64,
            completed_at_ms: Some(sequence as i64),
            kind: agent_protocol::ItemKind::SubAgent,
            status: ItemStatus::Completed,
            payload: ThreadItemPayload::SubAgent {
                child_thread_id: ThreadId::new("thread-child"),
                activity,
                detail: Some(format!("activity:{item_id}")),
            },
            metadata: json!({}),
        };
    let started = subagent_item("subagent-started", 3, 2, SubAgentActivityKind::Started);
    let interacted = subagent_item(
        "subagent-interacted",
        4,
        3,
        SubAgentActivityKind::Interacted,
    );
    let interrupted = subagent_item(
        "subagent-interrupted",
        5,
        4,
        SubAgentActivityKind::Interrupted,
    );
    block_on(projection_store.apply_history(ApplyThreadHistoryParams {
        session_id: thread.session_id.clone(),
        thread_id: thread.thread_id.clone(),
        changes: ThreadHistoryChangeSet {
            sequence: 5,
            changed_turns: vec![turn],
            changed_items: vec![item, started, interacted, interrupted],
            ..Default::default()
        },
    }))
    .expect("apply canonical history");
    drop(projection_store);
    let projection_store = ProjectionStore::initialize(database_path).expect("reopen store");

    let detail = block_on(runtime_session_read_detail_from_thread_store(
        &stored,
        ReadDetailOptions::default(),
        &[],
        &projection_store,
    ))
    .expect("canonical detail");

    assert_eq!(detail["items"], detail["thread_read"]["thread_items"]);
    assert_eq!(
        detail["items"][0]["id"],
        "item_message-read-model-canonical"
    );
    assert_eq!(detail["items"][0]["type"], "agent_message");
    assert_eq!(detail["items"][0]["status"], "completed");
    assert_eq!(detail["items"][0]["text"], "canonical item");
    assert_eq!(detail["items"][0]["contentParts"][0]["type"], "text");
    assert_eq!(detail["items"][0]["contentParts"][1]["type"], "media");
    assert_eq!(
        detail["items"][0]["contentParts"][1]["reference"]["uri"],
        "sidecar://media/read-model"
    );
    assert_eq!(
        detail["items"][0]["contentParts"][1]["reference"]["mime_type"],
        "image/png"
    );
    assert_eq!(
        detail["items"][0]["contentParts"][1]["reference"]["source_path"],
        "/tmp/media/read-model.png"
    );
    assert_eq!(detail["items"][0]["started_at"], "1970-01-01T00:00:00.001Z");
    assert_ne!(detail["items"][0]["id"], "event-read-model-running");
    let activities = detail["items"]
        .as_array()
        .expect("items")
        .iter()
        .filter(|item| item["type"] == "subagent_activity")
        .map(|item| {
            assert_eq!(item["session_id"], "thread-child");
            assert_eq!(item["status"], "completed");
            (
                item["id"].as_str().expect("item id").to_string(),
                item["status_label"].as_str().expect("activity").to_string(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        activities,
        vec![
            ("item_subagent-started".to_string(), "started".to_string()),
            (
                "item_subagent-interacted".to_string(),
                "interacted".to_string()
            ),
            (
                "item_subagent-interrupted".to_string(),
                "interrupted".to_string()
            ),
        ]
    );
}
