use super::*;
use crate::ProjectionStore;
use app_server_protocol::{AgentSessionReadParams, AgentSessionReadResponse};
use std::fs;
use std::path::Path;
use std::sync::Arc;

fn current_import_core(root: &Path) -> RuntimeCore {
    RuntimeCore::default().with_projection_store(Arc::new(
        ProjectionStore::initialize(root.join("projection.sqlite"))
            .expect("canonical projection store"),
    ))
}

fn read_current_session(
    core: &RuntimeCore,
    params: AgentSessionReadParams,
) -> Result<AgentSessionReadResponse, RuntimeCoreError> {
    futures::executor::block_on(core.read_session_current(params))
}

#[test]
fn commit_preserves_codex_tool_command_and_patch_timeline() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rollout_path = temp.path().join("rollout-runtime-events.jsonl");
    fs::write(
        &rollout_path,
        [
            session_meta("thread-runtime-events"),
            event_user_message("run it"),
            response_user_message("run it"),
            response_reasoning("I need to run the project tests first."),
            response_function_call(
                "call_exec",
                "exec_command",
                serde_json::json!({"cmd": "npm test", "workdir": "/workspace/app"}),
            ),
            event_exec_approval_request("call_exec"),
            response_function_call_output(
                "call_exec",
                "Exit code: 0\nWall time: 0 seconds\nOutput:\nok",
            ),
            response_function_call(
                "call_read_file",
                "read_file",
                serde_json::json!({"path": "/workspace/app/docs/imported-preview.md"}),
            ),
            response_function_call_output("call_read_file", "imported preview"),
            response_function_call(
                "call_read_html",
                "read_file",
                serde_json::json!({"path": "/workspace/app/docs/imported-preview.html"}),
            ),
            response_function_call_output("call_read_html", "imported html preview"),
            response_function_call(
                "call_read_docx",
                "read_file",
                serde_json::json!({"path": "/workspace/app/docs/imported-preview.docx"}),
            ),
            response_function_call_output("call_read_docx", "imported docx preview"),
            response_web_search_call("call_search", "search_query"),
            event_web_search_end(
                "call_search",
                "search_query",
                Some("Imported Search Source"),
            ),
            event_patch_apply_end("call_patch", true),
            event_agent_message("done"),
            event_user_message("run it"),
            event_agent_message("done again"),
        ]
        .join("\n"),
    )
    .expect("write rollout");
    let core = current_import_core(temp.path());

    let response = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect("commit");

    assert_eq!(response.imported_turns, 2);
    let session_id = response.session.session_id.clone();
    let stored_event_types = {
        let state = core
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        state
            .sessions
            .get(&session_id)
            .expect("imported stored session")
            .events
            .iter()
            .map(|event| event.event_type.clone())
            .collect::<Vec<_>>()
    };
    assert!(stored_event_types
        .iter()
        .any(|event_type| event_type == "item.started"));
    assert!(stored_event_types
        .iter()
        .any(|event_type| event_type == "item.completed"));
    assert!(!stored_event_types.iter().any(|event_type| matches!(
        event_type.as_str(),
        "tool.started" | "tool.result" | "tool.failed"
    )));

    let read = read_current_session(
        &core,
        AgentSessionReadParams {
            session_id,
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        },
    )
    .expect("read imported session");
    assert_eq!(read.turns.len(), 2);
    let detail = read.detail.expect("detail");
    let messages = detail["messages"].as_array().expect("messages");
    assert_eq!(
        messages
            .iter()
            .filter(|message| message["role"] == "user")
            .count(),
        2
    );

    let thread_read = &detail["thread_read"];
    let tool_calls = thread_read["tool_calls"].as_array().expect("tool calls");
    assert!(!tool_calls.iter().any(|tool| tool["id"] == "call_exec"));
    let commands = thread_read["commands"].as_array().expect("commands");
    assert!(commands.iter().any(|command| {
        command["command_id"] == "call_exec"
            && command["canonical_command"] == "npm test"
            && command["status"] == "completed"
    }));
    assert_eq!(thread_read["change_summary"]["applied_patch_count"], 1);
    let changed_files = thread_read["change_summary"]["changed_files"]
        .as_array()
        .expect("changed files");
    assert!(changed_files
        .iter()
        .any(|path| path == "/workspace/app/src/lib.rs"));
    assert!(changed_files
        .iter()
        .any(|path| path == "/workspace/app/docs/imported-preview.md"));

    let items = detail["items"].as_array().expect("timeline items");
    let first_turn_id = messages[1]["runtimeTurnId"]
        .as_str()
        .expect("first assistant runtime turn id");
    assert_eq!(
        items
            .iter()
            .filter(|item| {
                item["type"] == "command_execution" && item["command_id"] == "call_exec"
            })
            .count(),
        1
    );
    assert!(items.iter().any(|item| {
        item["type"] == "reasoning"
            && item["turn_id"] == first_turn_id
            && item["text"] == "I need to run the project tests first."
    }));
    assert!(items.iter().any(|item| {
        item["type"] == "command_execution"
            && item["turn_id"] == first_turn_id
            && item["command"] == "npm test"
            && item["aggregated_output"]
                .as_str()
                .is_some_and(|output| output.contains("ok"))
    }));
    assert!(items.iter().any(|item| {
        item["type"] == "patch"
            && item["turn_id"] == first_turn_id
            && item["paths"]
                .as_array()
                .is_some_and(|paths| paths.iter().any(|path| path == "/workspace/app/src/lib.rs"))
            && item["file_status"] == "applied"
            && item["status"] == "completed"
    }));
    assert!(items.iter().any(|item| {
        item["type"] == "tool_call"
            && item["turn_id"] == first_turn_id
            && item["call_id"] == "call_read_file"
            && item["tool_name"] == "read_file"
            && tool_argument_value(item, "path") == Some("/workspace/app/docs/imported-preview.md")
            && item["output"] == "imported preview"
    }));
    assert!(items.iter().any(|item| {
        item["type"] == "tool_call"
            && item["turn_id"] == first_turn_id
            && item["call_id"] == "call_read_html"
            && item["tool_name"] == "read_file"
            && tool_argument_value(item, "path")
                == Some("/workspace/app/docs/imported-preview.html")
            && item["output"] == "imported html preview"
    }));
    assert!(items.iter().any(|item| {
        item["type"] == "tool_call"
            && item["turn_id"] == first_turn_id
            && item["call_id"] == "call_read_docx"
            && item["tool_name"] == "read_file"
            && tool_argument_value(item, "path")
                == Some("/workspace/app/docs/imported-preview.docx")
            && item["output"] == "imported docx preview"
    }));
    assert!(items.iter().any(|item| {
        item["type"] == "web_search"
            && item["turn_id"] == first_turn_id
            && item["call_id"] == "call_search"
            && item["action"] == "search_query"
            && item["query"] == "imported query"
            && tool_argument_value(item, "query") == Some("imported query")
            && item["output"]
                .as_str()
                .is_some_and(|output| output.contains("Imported Search Source"))
            && item["status"] == "completed"
    }));
    assert!(items.iter().any(|item| {
        item["type"] == "approval_request"
            && item["turn_id"] == first_turn_id
            && item["request_id"] == "call_exec"
            && item["status"] == "completed"
            && item.get("response").is_none()
            && item["metadata"]["imported_read_only"] == true
    }));
    assert!(items.iter().any(|item| {
        item["type"] == "agent_message"
            && item["turn_id"] == first_turn_id
            && item["text"] == "done"
    }));
    assert!(read.turns.iter().any(|turn| turn.turn_id == first_turn_id));
}

#[test]
fn commit_preserves_high_volume_codex_tool_events_in_canonical_projection() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rollout_path = temp.path().join("rollout-runtime-event-budget.jsonl");
    let mut lines = vec![
        session_meta("thread-runtime-event-budget"),
        event_user_message("big run"),
    ];
    for index in 0..90 {
        let call_id = format!("call_exec_{index}");
        lines.push(response_function_call(
            &call_id,
            "exec_command",
            serde_json::json!({"cmd": format!("echo {index}")}),
        ));
        lines.push(response_function_call_output(
            &call_id,
            "Exit code: 0\nWall time: 0 seconds\nOutput:\nok",
        ));
    }
    lines.push(event_patch_apply_end("call_patch_budget", true));
    lines.push(event_agent_message("done"));
    fs::write(&rollout_path, lines.join("\n")).expect("write rollout");
    let core = current_import_core(temp.path());

    let response = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect("commit");

    assert_eq!(response.imported_turns, 1);
    assert!(!response
        .warnings
        .iter()
        .any(|warning| warning.contains("high-volume local history runtime events")));
    assert_eq!(response.summary.fidelity.budget_dropped, 0);
    assert_eq!(response.summary.fidelity.commands, 90);
    assert_eq!(response.summary.fidelity.patches, 1);
    let metadata = response
        .session
        .business_object_ref
        .as_ref()
        .and_then(|reference| reference.metadata.as_ref())
        .expect("import provenance metadata");
    assert!(metadata.get("importedRuntimeProjection").is_none());
    let session_id = response.session.session_id.clone();

    let read = read_current_session(
        &core,
        AgentSessionReadParams {
            session_id: session_id.clone(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        },
    )
    .expect("read imported session");
    let detail = read.detail.expect("detail");
    let thread_read = &detail["thread_read"];
    assert_eq!(
        thread_read["commands"].as_array().expect("commands").len(),
        90
    );
    assert_eq!(thread_read["change_summary"]["applied_patch_count"], 1);
    let items = detail["items"].as_array().expect("timeline items");
    assert_eq!(
        items
            .iter()
            .filter(|item| item["type"] == "command_execution")
            .count(),
        90
    );
    assert!(items
        .iter()
        .any(|item| item["type"] == "command_execution" && item["command_id"] == "call_exec_89"));
}

#[test]
fn commit_preserves_imported_commands_across_turns_without_projection_budget() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rollout_path = temp
        .path()
        .join("rollout-runtime-event-thread-budget.jsonl");
    let mut lines = vec![
        session_meta("thread-runtime-event-thread-budget"),
        event_user_message("first big run"),
    ];
    for index in 0..60 {
        let call_id = format!("call_exec_first_{index}");
        lines.push(response_function_call(
            &call_id,
            "exec_command",
            serde_json::json!({"cmd": format!("echo first {index}")}),
        ));
        lines.push(response_function_call_output(
            &call_id,
            "Exit code: 0\nWall time: 0 seconds\nOutput:\nok",
        ));
    }
    lines.push(event_agent_message("first done"));
    lines.push(event_user_message("second big run"));
    for index in 0..60 {
        let call_id = format!("call_exec_second_{index}");
        lines.push(response_function_call(
            &call_id,
            "exec_command",
            serde_json::json!({"cmd": format!("echo second {index}")}),
        ));
        lines.push(response_function_call_output(
            &call_id,
            "Exit code: 0\nWall time: 0 seconds\nOutput:\nok",
        ));
    }
    lines.push(event_agent_message("second done"));
    fs::write(&rollout_path, lines.join("\n")).expect("write rollout");
    let core = current_import_core(temp.path());

    let response = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect("commit");

    assert_eq!(response.imported_turns, 2);
    assert_eq!(response.summary.fidelity.commands, 120);
    let metadata = response
        .session
        .business_object_ref
        .as_ref()
        .and_then(|reference| reference.metadata.as_ref())
        .expect("import provenance metadata");
    assert!(metadata.get("importedRuntimeProjection").is_none());

    let session_id = response.session.session_id.clone();
    let read = read_current_session(
        &core,
        AgentSessionReadParams {
            session_id: session_id.clone(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        },
    )
    .expect("read imported session");
    let detail = read.detail.expect("detail");
    let thread_read = &detail["thread_read"];
    assert_eq!(
        thread_read["commands"].as_array().expect("commands").len(),
        120
    );
    assert!(thread_read["commands"]
        .as_array()
        .expect("commands")
        .iter()
        .any(|command| command["command_id"] == "call_exec_second_59"));
    assert!(thread_read["commands"]
        .as_array()
        .expect("commands")
        .iter()
        .any(|command| command["command_id"] == "call_exec_first_59"));
}

#[test]
fn commit_preserves_imported_assistant_message_order_between_runtime_events() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rollout_path = temp.path().join("rollout-runtime-event-order.jsonl");
    fs::write(
        &rollout_path,
        [
            session_meta("thread-runtime-event-order"),
            event_user_message("inspect first"),
            event_agent_message("我先说明第一步。"),
            response_reasoning("Need to inspect the project."),
            response_function_call(
                "call_exec_order",
                "exec_command",
                serde_json::json!({"cmd": "ls", "workdir": "/workspace/app"}),
            ),
            response_function_call_output(
                "call_exec_order",
                "Exit code: 0\nWall time: 0 seconds\nOutput:\nCargo.toml",
            ),
            event_agent_message("命令执行完成。"),
        ]
        .join("\n"),
    )
    .expect("write rollout");
    let core = current_import_core(temp.path());

    let response = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect("commit");

    let read = read_current_session(
        &core,
        AgentSessionReadParams {
            session_id: response.session.session_id,
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        },
    )
    .expect("read imported session");
    let detail = read.detail.expect("detail");
    let items = detail["items"].as_array().expect("timeline items");
    let ordered_types_and_text = items
        .iter()
        .map(|item| {
            (
                item["type"].as_str().unwrap_or_default().to_string(),
                item["text"]
                    .as_str()
                    .or_else(|| item["command"].as_str())
                    .unwrap_or_default()
                    .to_string(),
            )
        })
        .collect::<Vec<_>>();
    let first_assistant_index = ordered_types_and_text
        .iter()
        .position(|(kind, text)| kind == "agent_message" && text == "我先说明第一步。")
        .expect("first assistant message item");
    let reasoning_index = ordered_types_and_text
        .iter()
        .position(|(kind, text)| kind == "reasoning" && text == "Need to inspect the project.")
        .expect("reasoning item");
    let command_index = ordered_types_and_text
        .iter()
        .position(|(kind, text)| kind == "command_execution" && text == "ls")
        .expect("command item");
    let second_assistant_index = ordered_types_and_text
        .iter()
        .position(|(kind, text)| kind == "agent_message" && text == "命令执行完成。")
        .expect("second assistant message item");

    assert!(first_assistant_index < reasoning_index);
    assert!(reasoning_index < command_index);
    assert!(command_index < second_assistant_index);
    assert!(items.iter().any(|item| {
        item["type"] == "command_execution"
            && item["command_id"] == "call_exec_order"
            && item["status"] == "completed"
            && item["exit_code"] == 0
    }));
    assert!(!items.iter().any(|item| {
        item["type"] == "command_execution"
            && item["command_id"] == "call_exec_order"
            && item["status"] == "failed"
    }));
}

#[tokio::test]
async fn commit_delays_imported_turn_terminal_until_after_late_runtime_events() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rollout_path = temp
        .path()
        .join("rollout-late-runtime-after-terminal.jsonl");
    fs::write(
        &rollout_path,
        [
            session_meta("thread-late-runtime-after-terminal"),
            event_user_message("run after terminal"),
            event_turn_complete("turn-source-1"),
            response_function_call(
                "call_late_exec",
                "exec_command",
                serde_json::json!({"cmd": "echo late", "workdir": "/workspace/app"}),
            ),
            response_function_call_output(
                "call_late_exec",
                "Exit code: 0\nWall time: 0 seconds\nOutput:\nlate",
            ),
            event_agent_message("late answer"),
        ]
        .join("\n"),
    )
    .expect("write rollout");
    let core = current_import_core(temp.path());

    let response = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect("commit");

    assert_eq!(response.imported_turns, 1);
    let session_id = response.session.session_id.clone();
    let read = read_current_session(
        &core,
        AgentSessionReadParams {
            session_id: session_id.clone(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        },
    )
    .expect("read imported session");
    let detail = read.detail.expect("detail");
    assert_eq!(
        detail["turns"][0]["startedAt"], "2026-06-16T00:00:01.000Z",
        "imported turn must preserve the source start timestamp"
    );
    assert_eq!(
        detail["turns"][0]["completedAt"], "2026-06-16T00:00:01.375Z",
        "imported turn must not mix the source start timestamp with the import commit time"
    );
    let items = detail["items"].as_array().expect("timeline items");
    assert!(items.iter().any(|item| {
        item["type"] == "command_execution"
            && item["command_id"] == "call_late_exec"
            && item["status"] == "completed"
            && item["aggregated_output"]
                .as_str()
                .is_some_and(|output| output.contains("late"))
    }));
    assert!(items
        .iter()
        .any(|item| { item["type"] == "agent_message" && item["text"] == "late answer" }));

    let stored_events = core
        .state
        .lock()
        .expect("runtime core state mutex poisoned")
        .sessions
        .get(&session_id)
        .expect("imported stored session")
        .events
        .iter()
        .map(|event| (event.event_type.clone(), event.payload.clone()))
        .collect::<Vec<_>>();
    let terminal_index = stored_events
        .iter()
        .rposition(|(event_type, _)| event_type == "turn.completed")
        .expect("turn completed event");
    let command_item_id = agent_protocol::ItemId::new("codex-command-call_late_exec").to_string();
    let command_index = stored_events
        .iter()
        .position(|(event_type, payload)| {
            event_type == "item.started" && payload["item"]["itemId"] == command_item_id.as_str()
        })
        .expect("canonical command item started event");
    let message_index = stored_events
        .iter()
        .rposition(|(event_type, _)| event_type == "message.delta")
        .expect("assistant message event");
    assert!(command_index < terminal_index);
    assert!(message_index < terminal_index);
    assert_eq!(terminal_index, stored_events.len() - 1);
}

#[test]
fn commit_synthetic_terminal_uses_last_source_event_timestamp() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rollout_path = temp.path().join("rollout-synthetic-terminal-time.jsonl");
    fs::write(
        &rollout_path,
        [
            session_meta("thread-synthetic-terminal-time"),
            event_user_message("preserve source timing"),
            event_agent_message("done"),
        ]
        .join("\n"),
    )
    .expect("write rollout");
    let core = current_import_core(temp.path());

    let response = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect("commit");
    let read = read_current_session(
        &core,
        AgentSessionReadParams {
            session_id: response.session.session_id,
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        },
    )
    .expect("read imported session");
    let detail = read.detail.expect("detail");

    assert_eq!(detail["turns"][0]["startedAt"], "2026-06-16T00:00:01.000Z");
    assert_eq!(
        detail["turns"][0]["completedAt"],
        "2026-06-16T00:00:02.000Z"
    );
}

#[test]
fn commit_preserves_imported_update_plan_timeline_item() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rollout_path = temp.path().join("rollout-update-plan.jsonl");
    fs::write(
        &rollout_path,
        [
            session_meta("thread-update-plan"),
            event_user_message("plan the work"),
            response_reasoning("Need to create a concise checklist."),
            response_function_call(
                "call_update_plan",
                "update_plan",
                serde_json::json!({
                    "explanation": "Imported planning checkpoint",
                    "plan": [
                        {"step": "Inspect imported timeline", "status": "completed"},
                        {"step": "Project plan item", "status": "in_progress"},
                        {"step": "Verify rendering order", "status": "pending"}
                    ]
                }),
            ),
            event_agent_message("Plan captured."),
        ]
        .join("\n"),
    )
    .expect("write rollout");
    let core = current_import_core(temp.path());

    let response = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect("commit");

    let read = read_current_session(
        &core,
        AgentSessionReadParams {
            session_id: response.session.session_id,
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        },
    )
    .expect("read imported session");
    let detail = read.detail.expect("detail");
    let items = detail["items"].as_array().expect("timeline items");
    let plan = items
        .iter()
        .find(|item| item["type"] == "plan")
        .expect("plan item");

    assert_eq!(plan["status"], "completed");
    assert!(plan["text"]
        .as_str()
        .is_some_and(|text| text.contains("Project plan item")));
    assert_eq!(plan["metadata"]["source_client"], "codex");
    assert_eq!(plan["metadata"]["tool_call_id"], "call_update_plan");
    assert_eq!(
        plan["metadata"]["explanation"],
        "Imported planning checkpoint"
    );
    assert_eq!(
        plan["metadata"]["plan"][0]["step"],
        "Inspect imported timeline"
    );
    assert_eq!(plan["metadata"]["plan"][1]["status"], "in_progress");

    let ordered_types = items
        .iter()
        .map(|item| item["type"].as_str().unwrap_or_default())
        .collect::<Vec<_>>();
    let reasoning_index = ordered_types
        .iter()
        .position(|kind| *kind == "reasoning")
        .expect("reasoning item");
    let plan_index = ordered_types
        .iter()
        .position(|kind| *kind == "plan")
        .expect("plan item");
    assert!(!items
        .iter()
        .any(|item| item["type"] == "tool_call" && item["call_id"] == "call_update_plan"));
    assert!(reasoning_index < plan_index);
}

#[tokio::test]
async fn commit_imports_user_and_agent_items_with_canonical_lifecycle() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rollout_path = temp.path().join("rollout-message-lifecycle.jsonl");
    fs::write(
        &rollout_path,
        [
            session_meta("thread-message-lifecycle"),
            event_user_message("explain the result"),
            response_reasoning("I should verify the ordering first."),
            response_agent_message("msg_agent_4", "The verified result is ready."),
        ]
        .join("\n"),
    )
    .expect("write rollout");
    let projection_store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store"),
    );
    let core = RuntimeCore::default().with_projection_store(projection_store);

    let response = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect("commit");
    let session_id = response.session.session_id.clone();
    let stored_events = core.events_for_session(&session_id).expect("stored events");
    let user_item_id = agent_protocol::ItemId::new("imported-user_2").to_string();
    let agent_item_id = agent_protocol::ItemId::new("msg_agent_4").to_string();
    for item_id in [&user_item_id, &agent_item_id] {
        assert!(stored_events.iter().any(|event| {
            event.event_type == "item.started"
                && event.payload["item"]["itemId"] == item_id.as_str()
        }));
        assert!(stored_events.iter().any(|event| {
            event.event_type == "item.completed"
                && event.payload["item"]["itemId"] == item_id.as_str()
        }));
    }
    let current = core
        .read_session_current(AgentSessionReadParams {
            session_id,
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .await
        .expect("read canonical imported session");
    let detail = current.detail.expect("canonical detail");
    let items = detail["thread_read"]["thread_items"]
        .as_array()
        .expect("canonical thread items");
    let user = items
        .iter()
        .find(|item| item["id"].as_str() == Some(user_item_id.as_str()))
        .expect("canonical user item");
    let reasoning = items
        .iter()
        .find(|item| item["type"] == "reasoning")
        .expect("canonical reasoning item");
    let agent = items
        .iter()
        .find(|item| item["id"].as_str() == Some(agent_item_id.as_str()))
        .expect("canonical agent item");

    assert_eq!(user["type"], "user_message");
    assert_eq!(user["status"], "completed");
    assert_eq!(agent["type"], "agent_message");
    assert_eq!(agent["status"], "completed");
    let user_ordinal = user["ordinal"].as_u64().expect("user ordinal");
    let reasoning_ordinal = reasoning["ordinal"].as_u64().expect("reasoning ordinal");
    let agent_ordinal = agent["ordinal"].as_u64().expect("agent ordinal");
    assert!(user_ordinal < reasoning_ordinal);
    assert!(reasoning_ordinal < agent_ordinal);
}

#[tokio::test]
async fn commit_pairs_adjacent_codex_message_sources_without_losing_phase_or_item_id() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rollout_path = temp.path().join("rollout-message-source-pairs.jsonl");
    fs::write(
        &rollout_path,
        [
            session_meta("thread-message-source-pairs"),
            event_user_message("inspect the paired messages"),
            response_user_message("inspect the paired messages"),
            event_agent_message_with_phase("I will inspect the project first.", "commentary"),
            response_agent_message_with_phase(
                "msg_commentary_pair",
                "I will inspect the project first.",
                "commentary",
            ),
            event_agent_message_with_phase("The inspection is complete.", "final_answer"),
            response_agent_message_with_phase(
                "msg_final_pair",
                "The inspection is complete.",
                "final_answer",
            ),
        ]
        .join("\n"),
    )
    .expect("write rollout");
    let core = current_import_core(temp.path());

    let response = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect("commit");
    let current = core
        .read_session_current(AgentSessionReadParams {
            session_id: response.session.session_id,
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .await
        .expect("read canonical imported session");
    let detail = current.detail.expect("canonical detail");
    let agents = detail["items"]
        .as_array()
        .expect("canonical items")
        .iter()
        .filter(|item| item["type"] == "agent_message")
        .collect::<Vec<_>>();

    assert_eq!(
        agents.len(),
        2,
        "paired Codex sources must not duplicate text"
    );
    assert_eq!(agents[0]["text"], "I will inspect the project first.");
    assert_eq!(agents[0]["phase"], "commentary");
    assert_eq!(
        agents[0]["id"],
        agent_protocol::ItemId::new("msg_commentary_pair").to_string()
    );
    assert_eq!(agents[1]["text"], "The inspection is complete.");
    assert_eq!(agents[1]["phase"], "final_answer");
    assert_eq!(
        agents[1]["id"],
        agent_protocol::ItemId::new("msg_final_pair").to_string()
    );
}

#[test]
fn commit_preserves_intentionally_repeated_same_source_assistant_messages() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rollout_path = temp.path().join("rollout-repeated-assistant-message.jsonl");
    fs::write(
        &rollout_path,
        [
            session_meta("thread-repeated-assistant-message"),
            event_user_message("repeat the status exactly"),
            event_agent_message_with_phase("Still working.", "commentary"),
            event_agent_message_with_phase("Still working.", "commentary"),
            event_agent_message_with_phase("Done.", "final_answer"),
        ]
        .join("\n"),
    )
    .expect("write rollout");
    let core = current_import_core(temp.path());

    let response = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect("commit");
    let read = read_current_session(
        &core,
        AgentSessionReadParams {
            session_id: response.session.session_id,
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        },
    )
    .expect("read imported session");
    let detail = read.detail.expect("detail");
    let repeated_count = detail["items"]
        .as_array()
        .expect("canonical items")
        .iter()
        .filter(|item| item["type"] == "agent_message" && item["text"] == "Still working.")
        .count();

    assert_eq!(
        repeated_count, 2,
        "same-source repetition is user-visible content"
    );
}

#[tokio::test]
async fn commit_avoids_source_and_runtime_ordinal_collision() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rollout_path = temp.path().join("rollout-ordinal-collision.jsonl");
    fs::write(
        &rollout_path,
        [
            session_meta("thread-ordinal-collision"),
            event_user_message("first turn"),
            unsupported_rollout_item("turn_context"),
            unsupported_rollout_item("turn_context"),
            unsupported_rollout_item("turn_context"),
            response_function_call(
                "call_collision",
                "read_file",
                serde_json::json!({"path": "/workspace/app/README.md"}),
            ),
            response_function_call_output("call_collision", "read"),
            event_agent_message("first answer"),
            event_user_message("second turn"),
            event_agent_message("second answer"),
        ]
        .join("\n"),
    )
    .expect("write rollout");
    let projection_store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store"),
    );
    let core = RuntimeCore::default().with_projection_store(projection_store);

    let response = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect("commit");
    let current = core
        .read_session_current(AgentSessionReadParams {
            session_id: response.session.session_id,
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .await
        .expect("read canonical imported session after ordinal collision candidate");
    let detail = current.detail.expect("canonical detail");
    let items = detail["thread_read"]["thread_items"]
        .as_array()
        .expect("canonical thread items");
    let ordinals = items
        .iter()
        .map(|item| item["ordinal"].as_u64().expect("canonical ordinal"))
        .collect::<std::collections::BTreeSet<_>>();

    assert_eq!(
        ordinals.len(),
        items.len(),
        "canonical ordinals must be unique"
    );
    let tool = items
        .iter()
        .find(|item| item["call_id"] == "call_collision")
        .expect("imported tool");
    let user = items
        .iter()
        .find(|item| item["id"] == "imported-user_9")
        .expect("imported user");
    let agent = items
        .iter()
        .find(|item| item["id"] == "imported-agent_10")
        .expect("imported agent");
    assert_eq!(user["status"], "completed");
    assert_eq!(agent["status"], "completed");
    assert!(
        tool["ordinal"].as_u64().expect("tool ordinal")
            < user["ordinal"].as_u64().expect("user ordinal")
    );
    assert!(
        user["ordinal"].as_u64().expect("user ordinal")
            < agent["ordinal"].as_u64().expect("agent ordinal")
    );
}

#[test]
fn legacy_import_repair_uses_one_event_sequence_ordinal_domain() {
    let events = vec![
        app_server_protocol::AgentEvent {
            event_id: "legacy-tool-start".to_string(),
            sequence: 59,
            session_id: "session-legacy".to_string(),
            thread_id: Some("thread-legacy".to_string()),
            turn_id: Some("turn-1".to_string()),
            event_type: "item.started".to_string(),
            timestamp: "2026-07-14T00:00:01Z".to_string(),
            payload: serde_json::json!({
                "imported": true,
                "sourceClient": "codex",
                "item": {
                    "sessionId": "session-legacy",
                    "threadId": "thread-legacy",
                    "turnId": "turn-1",
                    "itemId": "imported-tool-call_collision",
                    "sequence": 0,
                    "ordinal": 126,
                    "createdAtMs": 0,
                    "updatedAtMs": 0,
                    "completedAtMs": null,
                    "kind": "tool",
                    "status": "inProgress",
                    "payload": {
                        "type": "tool",
                        "call_id": "call_collision",
                        "name": "read_file",
                        "arguments": [],
                        "output": null
                    },
                    "metadata": {}
                }
            }),
        },
        app_server_protocol::AgentEvent {
            event_id: "legacy-user".to_string(),
            sequence: 126,
            session_id: "session-legacy".to_string(),
            thread_id: Some("thread-legacy".to_string()),
            turn_id: Some("turn-2".to_string()),
            event_type: "message.created".to_string(),
            timestamp: "2026-07-14T00:00:02Z".to_string(),
            payload: serde_json::json!({
                "role": "user",
                "input": [{"type": "text", "text": "continue"}]
            }),
        },
        app_server_protocol::AgentEvent {
            event_id: "legacy-reasoning".to_string(),
            sequence: 127,
            session_id: "session-legacy".to_string(),
            thread_id: Some("thread-legacy".to_string()),
            turn_id: Some("turn-2".to_string()),
            event_type: "reasoning.completed".to_string(),
            timestamp: "2026-07-14T00:00:03Z".to_string(),
            payload: serde_json::json!({
                "imported": true,
                "sourceClient": "codex",
                "text": "inspect ordering"
            }),
        },
        app_server_protocol::AgentEvent {
            event_id: "legacy-agent".to_string(),
            sequence: 128,
            session_id: "session-legacy".to_string(),
            thread_id: Some("thread-legacy".to_string()),
            turn_id: Some("turn-2".to_string()),
            event_type: "message.delta".to_string(),
            timestamp: "2026-07-14T00:00:04Z".to_string(),
            payload: serde_json::json!({
                "imported": true,
                "sourceClient": "codex",
                "role": "assistant",
                "text": "ordered answer"
            }),
        },
    ];

    let changes = crate::runtime::thread_item_projection::materialize_events(
        &events,
        "session-legacy",
        "thread-legacy",
    )
    .expect("materialize legacy import repair");
    let tool = changes
        .changed_items
        .iter()
        .find(|item| item.item_id.as_str() == "imported-tool-call_collision")
        .expect("legacy tool item");
    let user = changes
        .changed_items
        .iter()
        .find(|item| {
            matches!(
                item.payload,
                agent_protocol::ThreadItemPayload::UserMessage { .. }
            )
        })
        .expect("legacy user item");
    let reasoning = changes
        .changed_items
        .iter()
        .find(|item| {
            matches!(
                item.payload,
                agent_protocol::ThreadItemPayload::Reasoning { .. }
            )
        })
        .expect("legacy reasoning item");
    let agent = changes
        .changed_items
        .iter()
        .find(|item| {
            matches!(
                item.payload,
                agent_protocol::ThreadItemPayload::AgentMessage { .. }
            )
        })
        .expect("legacy agent item");

    assert_eq!(tool.ordinal, 59);
    assert_eq!(user.ordinal, 126);
    assert_eq!(reasoning.ordinal, 127);
    assert_eq!(agent.ordinal, 128);
    assert_eq!(agent.status, agent_protocol::ItemStatus::Completed);
    assert!(agent.completed_at_ms.is_some());
    assert_ne!(tool.ordinal, user.ordinal);
}

#[tokio::test]
async fn commit_preserves_imported_completed_plan_item() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rollout_path = temp.path().join("rollout-completed-plan-item.jsonl");
    fs::write(
        &rollout_path,
        [
            session_meta("thread-completed-plan-item"),
            event_user_message("draft a plan"),
            event_completed_plan_item(
                "turn-plan-item",
                "item-plan-1",
                "# Final plan\n- first\n- second\n",
            ),
            event_agent_message("Plan ready."),
        ]
        .join("\n"),
    )
    .expect("write rollout");
    let projection_store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store"),
    );
    let core = RuntimeCore::default().with_projection_store(projection_store);

    let response = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect("commit");

    let read_params = AgentSessionReadParams {
        session_id: response.session.session_id,
        history_limit: None,
        history_offset: None,
        history_before_message_id: None,
    };
    let presentation_read = core
        .read_session(read_params.clone())
        .expect("read imported presentation");
    let canonical_read = core
        .read_session_current(read_params)
        .await
        .expect("read imported canonical session");
    let presentation_detail = presentation_read.detail.expect("presentation detail");
    let canonical_detail = canonical_read.detail.expect("canonical detail");
    let items = presentation_detail["items"]
        .as_array()
        .expect("timeline items");
    let plan = items
        .iter()
        .find(|item| item["type"] == "plan")
        .expect("plan item");
    let canonical_plan = canonical_detail["thread_read"]["thread_items"]
        .as_array()
        .expect("canonical thread items")
        .iter()
        .find(|item| item["type"] == "plan")
        .expect("canonical plan item");

    assert_eq!(plan["status"], "completed");
    assert_eq!(plan["text"], "# Final plan\n- first\n- second");
    assert_eq!(plan["metadata"]["source_client"], "codex");
    assert_eq!(plan["metadata"]["source_item_id"], "item-plan-1");
    assert_eq!(
        plan["metadata"]["source_provenance"]["sourcePayloadType"],
        "item_completed"
    );
    assert_eq!(canonical_plan["id"], "item-plan-1");
    assert_eq!(canonical_plan["status"], "completed");
    assert_eq!(canonical_plan["metadata"]["revisionId"], "item-plan-1");
    assert!(items
        .iter()
        .any(|item| item["type"] == "agent_message" && item["text"] == "Plan ready."));
}

#[test]
fn commit_projects_codex_runtime_specialized_items_into_existing_timeline_types() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rollout_path = temp.path().join("rollout-specialized-runtime-items.jsonl");
    fs::write(
        &rollout_path,
        [
            session_meta("thread-specialized-runtime-items"),
            event_user_message("import specialized runtime items"),
            event_mcp_tool_call_begin("call_mcp"),
            event_mcp_tool_call_end("call_mcp"),
            event_dynamic_tool_call_request("call_dynamic"),
            event_dynamic_tool_call_response("call_dynamic"),
            event_view_image_tool_call("call_image_view"),
            event_image_generation_begin("call_image_gen"),
            event_image_generation_end("call_image_gen"),
            event_context_compacted(),
            event_entered_review_mode("Review current changes."),
            event_exited_review_mode("Review findings:\n- src/lib.rs: looks good"),
            event_subagent_activity("subagent-event-1"),
            event_collab_agent_spawn_begin("call_collab"),
            event_collab_agent_spawn_end("call_collab"),
            event_agent_message("Specialized items imported."),
        ]
        .join("\n"),
    )
    .expect("write rollout");
    let core = current_import_core(temp.path());

    let response = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect("commit");

    let read = read_current_session(
        &core,
        AgentSessionReadParams {
            session_id: response.session.session_id,
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        },
    )
    .expect("read imported session");
    let detail = read.detail.expect("detail");
    let items = detail["items"].as_array().expect("timeline items");
    assert!(items.iter().any(|item| {
        item["type"] == "tool_call"
            && item["call_id"] == "call_mcp"
            && item["mcp_server"] == "filesystem"
            && item["tool_name"] == "read_file"
            && item["status"] == "completed"
            && tool_argument_value(item, "path") == Some("src/lib.rs")
            && item["metadata"]["source_client"] == "codex"
    }));
    assert!(items.iter().any(|item| {
        item["type"] == "tool_call"
            && item["call_id"] == "call_dynamic"
            && item["tool_name"] == "docs.lookup"
            && item["output"] == "dynamic result"
    }));
    assert!(items.iter().any(|item| {
        item["type"] == "tool_call"
            && item["call_id"] == "call_image_view"
            && item["tool_name"] == "view_image"
            && tool_argument_value(item, "path") == Some("/workspace/app/assets/input.png")
    }));
    assert!(items.iter().any(|item| {
        item["type"] == "tool_call"
            && item["call_id"] == "call_image_gen"
            && item["tool_name"] == "image_generation"
            && item["output"] == "/workspace/app/assets/result.png"
    }));
    assert!(items.iter().any(|item| {
        item["type"] == "context_compaction"
            && item["status"] == "completed"
            && item["stage"] == "completed"
            && item["metadata"]["source_client"] == "codex"
    }));
    assert!(items.iter().any(|item| {
        item["type"] == "reasoning"
            && item["text"] == "Review current changes."
            && item["metadata"]["source_client"] == "codex"
    }));
    assert!(items.iter().any(|item| {
        item["type"] == "agent_message"
            && item["text"]
                .as_str()
                .is_some_and(|text| text.contains("Review findings"))
            && item["metadata"]["source_client"] == "codex"
    }));
    assert!(
        items.iter().any(|item| {
            item["type"] == "subagent_activity"
                && item["metadata"]["source_event_id"] == "subagent-event-1"
                && item["status_label"] == "started"
                && item["session_id"] == "subagent-thread-1"
        }),
        "missing canonical Codex subagent activity: {}",
        serde_json::to_string_pretty(items).expect("serialize timeline items")
    );
    assert!(items.iter().any(|item| {
        item["type"] == "tool_call"
            && item["call_id"] == "call_collab"
            && item["tool_name"] == "agent"
            && item["status"] == "completed"
            && item["output"] == "subagent-thread-2"
    }));
}

#[test]
fn commit_only_lowers_wait_agent_to_collab_agent_tool_call() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rollout_path = temp.path().join("rollout-collab-producer-boundary.jsonl");
    let non_wait_tools = [
        "spawn_agent",
        "send_message",
        "followup_task",
        "interrupt_agent",
        "resume_agent",
        "close_agent",
    ];
    let mut lines = vec![
        session_meta("thread-collab-producer-boundary"),
        event_user_message("exercise collaboration tools"),
        response_function_call("call_wait", "wait_agent", serde_json::json!({})),
        response_function_call_output("call_wait", "ready"),
    ];
    for tool_name in non_wait_tools {
        let call_id = format!("call_{tool_name}");
        lines.push(response_function_call(
            &call_id,
            tool_name,
            serde_json::json!({
                "target_thread_id": "thread-child",
                "message": "continue"
            }),
        ));
        lines.push(response_function_call_output(&call_id, "done"));
    }
    lines.push(event_agent_message("Collaboration tools imported."));
    fs::write(&rollout_path, lines.join("\n")).expect("write rollout");
    let core = current_import_core(temp.path());

    let response = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect("commit");

    let read = read_current_session(
        &core,
        AgentSessionReadParams {
            session_id: response.session.session_id,
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        },
    )
    .expect("read imported session");
    let detail = read.detail.expect("detail");
    let items = detail["thread_read"]["thread_items"]
        .as_array()
        .expect("canonical thread items");
    let wait = items
        .iter()
        .find(|item| item["call_id"] == "call_wait")
        .expect("wait agent item");
    assert_eq!(wait["id"], "codex-collab-call_wait");
    assert_eq!(wait["type"], "tool_call");
    assert_eq!(wait["tool_name"], "wait_agent");

    for tool_name in non_wait_tools {
        let call_id = format!("call_{tool_name}");
        let item = items
            .iter()
            .find(|item| item["call_id"] == call_id)
            .unwrap_or_else(|| panic!("missing ordinary tool item for {tool_name}"));
        assert_eq!(item["id"], format!("codex-tool-{call_id}"));
        assert_eq!(item["type"], "tool_call");
        assert_eq!(item["tool_name"], tool_name);
    }
}

#[test]
fn commit_merges_duplicate_user_messages_when_response_item_precedes_event_msg() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rollout_path = temp.path().join("rollout-duplicate-user-reversed.jsonl");
    fs::write(
        &rollout_path,
        [
            session_meta("thread-duplicate-user-reversed"),
            response_user_message("first user message"),
            event_user_message("first user message"),
            event_agent_message("first answer"),
        ]
        .join("\n"),
    )
    .expect("write rollout");
    let core = current_import_core(temp.path());

    let response = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect("commit");

    assert_eq!(response.imported_turns, 1);

    let read = core
        .read_session(AgentSessionReadParams {
            session_id: response.session.session_id,
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read imported session");
    let detail = read.detail.expect("detail");
    let messages = detail["messages"].as_array().expect("messages");
    assert_eq!(
        messages
            .iter()
            .filter(|message| message["role"] == "user")
            .count(),
        1
    );
    assert_eq!(messages[0]["content"][0]["text"], "first user message");
    assert_eq!(messages[0]["timestamp"], 1781568001);
    assert_eq!(messages[1]["content"][0]["text"], "first answer");
}

#[test]
fn commit_keeps_non_adjacent_duplicate_user_messages_as_separate_turns() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rollout_path = temp
        .path()
        .join("rollout-non-adjacent-duplicate-user.jsonl");
    fs::write(
        &rollout_path,
        [
            session_meta("thread-non-adjacent-duplicate-user"),
            response_user_message("same request"),
            response_reasoning("intervening source event"),
            event_user_message("same request"),
            event_agent_message("second answer"),
        ]
        .join("\n"),
    )
    .expect("write rollout");
    let core = current_import_core(temp.path());

    let response = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect("commit");

    assert_eq!(response.imported_turns, 2);
    let read = core
        .read_session(AgentSessionReadParams {
            session_id: response.session.session_id,
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read imported session");
    let detail = read.detail.expect("detail");
    let messages = detail["messages"].as_array().expect("messages");
    assert_eq!(
        messages
            .iter()
            .filter(|message| message["role"] == "user")
            .count(),
        2
    );
}

#[test]
fn commit_merges_image_user_sources_without_duplicate_turn_or_attachment() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rollout_path = temp.path().join("rollout-duplicate-image-user.jsonl");
    fs::write(
        &rollout_path,
        [
            session_meta("thread-duplicate-image-user"),
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:01.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "<image name=[Image #1] path=\"/tmp/photo.png\">\n</image>\n[Image #1] inspect this image"
                        },
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,aW1hZ2U="
                        }
                    ]
                }
            })
            .to_string(),
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:01.100Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": "## My request for Codex: [Image #1] inspect this image",
                    "local_images": ["/tmp/photo.png"]
                }
            })
            .to_string(),
            event_agent_message("image answer"),
        ]
        .join("\n"),
    )
    .expect("write rollout");
    let core = current_import_core(temp.path());

    let response = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect("commit");

    assert_eq!(response.imported_turns, 1);
    let read = read_current_session(
        &core,
        AgentSessionReadParams {
            session_id: response.session.session_id,
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        },
    )
    .expect("read imported session");
    let detail = read.detail.expect("detail");
    let messages = detail["messages"].as_array().expect("messages");
    assert_eq!(
        messages
            .iter()
            .filter(|message| message["role"] == "user")
            .count(),
        1
    );
    assert_eq!(
        messages[0]["content"][0]["text"],
        "[Image #1] inspect this image"
    );

    let items = detail["thread_read"]["thread_items"]
        .as_array()
        .expect("canonical thread items");
    let user = items
        .iter()
        .find(|item| item["type"] == "user_message")
        .expect("canonical user item");
    let image_count = user["content"]
        .as_array()
        .expect("user content")
        .iter()
        .filter(|part| part["type"] == "image")
        .count();
    assert_eq!(image_count, 1);
}

#[test]
fn commit_closes_incomplete_imported_lifecycles_as_failed_timeline_items() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rollout_path = temp.path().join("rollout-incomplete-lifecycle.jsonl");
    fs::write(
        &rollout_path,
        [
            session_meta("thread-incomplete-lifecycle"),
            event_user_message("run incomplete tool"),
            response_function_call(
                "call_incomplete",
                "exec_command",
                serde_json::json!({"cmd": "npm test", "workdir": "/workspace/app"}),
            ),
            response_function_call(
                "call_tool_incomplete",
                "read_file",
                serde_json::json!({"path": "src/lib.rs"}),
            ),
            event_patch_started("patch_incomplete"),
            event_agent_message("imported answer"),
        ]
        .join("\n"),
    )
    .expect("write rollout");
    let core = current_import_core(temp.path());

    let response = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect("commit");

    let read = read_current_session(
        &core,
        AgentSessionReadParams {
            session_id: response.session.session_id,
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        },
    )
    .expect("read imported session");
    let detail = read.detail.expect("detail");
    let items = detail["items"].as_array().expect("timeline items");
    let command = items
        .iter()
        .find(|item| item["type"] == "command_execution" && item["command_id"] == "call_incomplete")
        .expect("command item");
    assert_eq!(command["status"], "failed");
    assert_eq!(command["metadata"]["imported_incomplete"], true);
    let tool = items
        .iter()
        .find(|item| item["type"] == "tool_call" && item["call_id"] == "call_tool_incomplete")
        .expect("tool item");
    assert_eq!(tool["status"], "failed");
    assert_eq!(tool["metadata"]["imported_incomplete"], true);
    let patch = items
        .iter()
        .find(|item| item["type"] == "patch" && item["metadata"]["imported_incomplete"] == true)
        .expect("patch item");
    assert_eq!(patch["status"], "failed");
    assert_eq!(patch["metadata"]["imported_incomplete"], true);
    assert_eq!(
        items
            .iter()
            .filter(|item| item["status"] == "failed")
            .count(),
        3
    );
}

fn session_meta(thread_id: &str) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:00.000Z",
        "type": "session_meta",
        "payload": {
            "id": thread_id,
            "timestamp": "2026-06-16T00:00:00.000Z",
            "cwd": "/workspace/app",
            "source": "cli",
            "model_provider": "openai"
        }
    })
    .to_string()
}

fn event_user_message(message: &str) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:01.000Z",
        "type": "event_msg",
        "payload": {
            "type": "user_message",
            "message": format!("## My request for Codex: {message}")
        }
    })
    .to_string()
}

fn event_agent_message(message: &str) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:02.000Z",
        "type": "event_msg",
        "payload": {
            "type": "agent_message",
            "message": message
        }
    })
    .to_string()
}

fn event_agent_message_with_phase(message: &str, phase: &str) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:02.000Z",
        "type": "event_msg",
        "payload": {
            "type": "agent_message",
            "message": message,
            "phase": phase
        }
    })
    .to_string()
}

fn response_user_message(message: &str) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:01.100Z",
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": message}]
        }
    })
    .to_string()
}

fn response_agent_message(item_id: &str, message: &str) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:02.000Z",
        "type": "response_item",
        "payload": {
            "type": "message",
            "id": item_id,
            "role": "assistant",
            "content": [{"type": "output_text", "text": message}]
        }
    })
    .to_string()
}

fn response_agent_message_with_phase(item_id: &str, message: &str, phase: &str) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:02.000Z",
        "type": "response_item",
        "payload": {
            "type": "message",
            "id": item_id,
            "role": "assistant",
            "content": [{"type": "output_text", "text": message}],
            "phase": phase
        }
    })
    .to_string()
}

fn unsupported_rollout_item(item_type: &str) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:01.050Z",
        "type": item_type,
        "payload": {}
    })
    .to_string()
}

fn response_function_call(call_id: &str, name: &str, arguments: serde_json::Value) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:01.200Z",
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": arguments.to_string()
        }
    })
    .to_string()
}

fn response_reasoning(text: &str) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:01.150Z",
        "type": "response_item",
        "payload": {
            "type": "reasoning",
            "content": [{"type": "reasoning_text", "text": text}],
            "summary": [{"type": "summary_text", "text": text}]
        }
    })
    .to_string()
}

fn response_function_call_output(call_id: &str, output: &str) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:01.300Z",
        "type": "response_item",
        "payload": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output
        }
    })
    .to_string()
}

fn response_web_search_call(call_id: &str, action: &str) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:01.350Z",
        "type": "response_item",
        "payload": {
            "type": "web_search_call",
            "id": call_id,
            "status": "in_progress",
            "action": {
                "type": action,
                "query": "imported query"
            }
        }
    })
    .to_string()
}

fn event_web_search_end(call_id: &str, action: &str, title: Option<&str>) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:01.360Z",
        "type": "event_msg",
        "payload": {
            "type": "web_search_end",
            "call_id": call_id,
            "query": "imported query",
            "action": {
                "type": action
            },
            "results": [{
                "title": title.unwrap_or("Imported Search Result"),
                "url": "https://example.com/imported-search",
                "snippet": "Imported Codex search source"
            }]
        }
    })
    .to_string()
}

fn event_exec_approval_request(call_id: &str) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:01.370Z",
        "type": "event_msg",
        "payload": {
            "type": "exec_approval_request",
            "call_id": call_id,
            "command": ["npm", "test"],
            "reason": "Codex needs approval to run tests"
        }
    })
    .to_string()
}

fn event_turn_complete(turn_id: &str) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:01.375Z",
        "type": "event_msg",
        "payload": {
            "type": "turn_complete",
            "turn_id": turn_id,
            "completed_at": "2026-06-16T00:00:01.375Z",
            "duration_ms": 10
        }
    })
    .to_string()
}

fn event_completed_plan_item(turn_id: &str, item_id: &str, text: &str) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:01.380Z",
        "type": "event_msg",
        "payload": {
            "type": "item_completed",
            "thread_id": "thread-completed-plan-item",
            "turn_id": turn_id,
            "completed_at_ms": 1781568001380_i64,
            "item": {
                "type": "Plan",
                "id": item_id,
                "text": text
            }
        }
    })
    .to_string()
}

fn event_mcp_tool_call_begin(call_id: &str) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:01.381Z",
        "type": "event_msg",
        "payload": {
            "type": "mcp_tool_call_begin",
            "call_id": call_id,
            "invocation": {
                "server": "filesystem",
                "tool": "read_file",
                "arguments": {"path": "src/lib.rs"}
            }
        }
    })
    .to_string()
}

fn event_mcp_tool_call_end(call_id: &str) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:01.382Z",
        "type": "event_msg",
        "payload": {
            "type": "mcp_tool_call_end",
            "call_id": call_id,
            "invocation": {
                "server": "filesystem",
                "tool": "read_file",
                "arguments": {"path": "src/lib.rs"}
            },
            "result": {
                "Ok": {
                    "isError": false,
                    "content": [{"type": "text", "text": "file content"}]
                }
            }
        }
    })
    .to_string()
}

fn event_dynamic_tool_call_request(call_id: &str) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:01.383Z",
        "type": "event_msg",
        "payload": {
            "type": "dynamic_tool_call_request",
            "call_id": call_id,
            "namespace": "docs",
            "tool": "lookup",
            "arguments": {"query": "timeline"}
        }
    })
    .to_string()
}

fn event_dynamic_tool_call_response(call_id: &str) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:01.384Z",
        "type": "event_msg",
        "payload": {
            "type": "dynamic_tool_call_response",
            "call_id": call_id,
            "namespace": "docs",
            "tool": "lookup",
            "arguments": {"query": "timeline"},
            "success": true,
            "content_items": [{"type": "input_text", "text": "dynamic result"}]
        }
    })
    .to_string()
}

fn event_view_image_tool_call(call_id: &str) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:01.385Z",
        "type": "event_msg",
        "payload": {
            "type": "view_image_tool_call",
            "call_id": call_id,
            "path": "/workspace/app/assets/input.png"
        }
    })
    .to_string()
}

fn event_image_generation_begin(call_id: &str) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:01.386Z",
        "type": "event_msg",
        "payload": {
            "type": "image_generation_begin",
            "call_id": call_id,
            "prompt": "draw a timeline"
        }
    })
    .to_string()
}

fn event_image_generation_end(call_id: &str) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:01.387Z",
        "type": "event_msg",
        "payload": {
            "type": "image_generation_end",
            "call_id": call_id,
            "status": "completed",
            "result": "/workspace/app/assets/result.png",
            "saved_path": "/workspace/app/assets/result.png"
        }
    })
    .to_string()
}

fn event_context_compacted() -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:01.388Z",
        "type": "event_msg",
        "payload": {
            "type": "context_compacted"
        }
    })
    .to_string()
}

fn event_entered_review_mode(review: &str) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:01.389Z",
        "type": "event_msg",
        "payload": {
            "type": "entered_review_mode",
            "user_facing_hint": review
        }
    })
    .to_string()
}

fn event_exited_review_mode(review: &str) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:01.391Z",
        "type": "event_msg",
        "payload": {
            "type": "exited_review_mode",
            "review": review
        }
    })
    .to_string()
}

fn tool_argument_value<'a>(item: &'a serde_json::Value, name: &str) -> Option<&'a str> {
    item.get("arguments")?
        .as_array()?
        .iter()
        .find(|argument| argument.get("name").and_then(serde_json::Value::as_str) == Some(name))?
        .get("value")?
        .as_str()
}

fn event_subagent_activity(event_id: &str) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:01.392Z",
        "type": "event_msg",
        "payload": {
            "type": "sub_agent_activity",
            "event_id": event_id,
            "kind": "started",
            "agent_thread_id": "subagent-thread-1",
            "agent_path": "agents/reviewer.md"
        }
    })
    .to_string()
}

fn event_collab_agent_spawn_begin(call_id: &str) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:01.393Z",
        "type": "event_msg",
        "payload": {
            "type": "collab_agent_spawn_begin",
            "call_id": call_id,
            "sender_thread_id": "main-thread",
            "prompt": "review imported timeline",
            "model": "gpt-5.5"
        }
    })
    .to_string()
}

fn event_collab_agent_spawn_end(call_id: &str) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:01.394Z",
        "type": "event_msg",
        "payload": {
            "type": "collab_agent_spawn_end",
            "call_id": call_id,
            "sender_thread_id": "main-thread",
            "prompt": "review imported timeline",
            "model": "gpt-5.5",
            "new_thread_id": "subagent-thread-2",
            "status": "running"
        }
    })
    .to_string()
}

fn event_patch_apply_end(call_id: &str, success: bool) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:01.400Z",
        "type": "event_msg",
        "payload": {
            "type": "patch_apply_end",
            "call_id": call_id,
            "stdout": "Success. Updated files",
            "stderr": "",
            "success": success,
            "changes": {
                "/workspace/app/src/lib.rs": {"type": "modify"}
            }
        }
    })
    .to_string()
}

fn event_patch_started(call_id: &str) -> String {
    serde_json::json!({
        "timestamp": "2026-06-16T00:00:01.390Z",
        "type": "event_msg",
        "payload": {
            "type": "patch_apply_begin",
            "call_id": call_id,
            "changes": {
                "/workspace/app/src/lib.rs": {"type": "modify"}
            }
        }
    })
    .to_string()
}
