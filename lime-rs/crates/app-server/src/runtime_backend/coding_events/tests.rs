use super::*;
use agent_protocol::{ItemId, SessionId, ThreadId, TurnId};
use lime_agent::AgentToolResult;

fn success_result(output: &str, metadata: HashMap<String, Value>) -> AgentToolResult {
    AgentToolResult {
        success: true,
        output: output.to_string(),
        error: None,
        structured_content: None,
        images: None,
        metadata: Some(metadata),
    }
}

fn tool_started(tool_name: &str, tool_id: &str, arguments: Option<Value>) -> RuntimeAgentEvent {
    canonical_tool_event(tool_name, tool_id, arguments, None)
}

fn tool_completed(tool_id: &str, result: AgentToolResult) -> RuntimeAgentEvent {
    canonical_tool_event("tool", tool_id, None, Some(result))
}

fn canonical_tool_event(
    tool_name: &str,
    tool_id: &str,
    arguments: Option<Value>,
    result: Option<AgentToolResult>,
) -> RuntimeAgentEvent {
    let arguments = arguments.map(canonical_test_arguments).unwrap_or_default();
    let (status, output, metadata) = match result {
        Some(result) => {
            let status = if result.success {
                ItemStatus::Completed
            } else {
                ItemStatus::Failed
            };
            let metadata = result
                .metadata
                .map(|metadata| Value::Object(metadata.into_iter().collect()))
                .unwrap_or_else(|| json!({}));
            let output = ToolOutput {
                text: Some(result.output),
                structured_content: result.structured_content,
                error: result.error,
                duration_ms: metadata.get("duration_ms").and_then(Value::as_u64),
                truncated: metadata
                    .get("truncated")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                output_ref: metadata
                    .get("output_ref")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            };
            (status, Some(output), metadata)
        }
        None => (ItemStatus::InProgress, None, json!({})),
    };
    let payload = ThreadItemPayload::Tool {
        call_id: tool_id.to_string(),
        name: tool_name.to_string(),
        arguments,
        output,
    };
    let item = ThreadItem {
        session_id: SessionId::new("session-test"),
        thread_id: ThreadId::new("thread-test"),
        turn_id: TurnId::new("turn-test"),
        item_id: ItemId::new(tool_id),
        sequence: 1,
        ordinal: 1,
        created_at_ms: 1,
        updated_at_ms: 2,
        completed_at_ms: status.is_terminal().then_some(2),
        kind: payload.kind(),
        status,
        payload,
        metadata,
    };
    if status.is_terminal() {
        RuntimeAgentEvent::ItemCompleted { item }
    } else {
        RuntimeAgentEvent::ItemStarted { item }
    }
}

fn canonical_test_arguments(arguments: Value) -> Vec<ToolArgument> {
    match arguments {
        Value::Object(arguments) => arguments
            .into_iter()
            .map(|(name, value)| ToolArgument {
                name,
                value: value
                    .as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| value.to_string()),
            })
            .collect(),
        value => vec![ToolArgument {
            name: "value".to_string(),
            value: value
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| value.to_string()),
        }],
    }
}

#[test]
fn shell_tool_events_emit_command_and_test_lifecycle() {
    let mut mirror = CodingEventMirror::default();

    let started = mirror.process_event(&tool_started(
        "Bash",
        "tool-1",
        Some(json!({ "command": "cargo test -p app-server coding_events" })),
    ));
    let output = mirror.process_event(&RuntimeAgentEvent::ToolOutputDelta {
        tool_id: "tool-1".to_string(),
        delta: "running tests".to_string(),
        output_kind: Some("stdout".to_string()),
        metadata: None,
    });
    let ended = mirror.process_event(&tool_completed(
        "tool-1",
        success_result(
            "ok",
            HashMap::from([
                ("exit_code".to_string(), json!(0)),
                (
                    "command".to_string(),
                    json!("cargo test -p app-server coding_events"),
                ),
                ("cwd".to_string(), json!("/workspace")),
                ("shell".to_string(), json!("bash")),
            ]),
        ),
    ));

    let event_types = started
        .after_raw
        .into_iter()
        .chain(output.after_raw)
        .chain(ended.after_raw)
        .map(|event| event.event_type)
        .collect::<Vec<_>>();
    assert_eq!(
        event_types,
        vec![
            "command.started",
            "test.started",
            "command.output",
            "command.exited",
            "test.completed"
        ]
    );
}

#[test]
fn shell_tool_output_delta_preserves_process_lifecycle_metadata() {
    let mut mirror = CodingEventMirror::default();

    let _ = mirror.process_event(&tool_started(
        "Bash",
        "tool-process",
        Some(json!({ "command": "npm test" })),
    ));
    let output = mirror.process_event(&RuntimeAgentEvent::ToolOutputDelta {
        tool_id: "tool-process".to_string(),
        delta: "running".to_string(),
        output_kind: Some("stdout".to_string()),
        metadata: Some(HashMap::from([
            ("processId".to_string(), json!("process-tool-process")),
            ("executionProcessStatus".to_string(), json!("running")),
            ("outputBytes".to_string(), json!(7)),
            ("outputOmittedBytes".to_string(), json!(0)),
            ("outputTruncated".to_string(), json!(false)),
        ])),
    });

    assert_eq!(output.after_raw.len(), 1);
    assert_eq!(output.after_raw[0].event_type, "command.output");
    assert_eq!(
        output.after_raw[0].payload["metadata"]["processId"].as_str(),
        Some("process-tool-process")
    );
    assert_eq!(
        output.after_raw[0].payload["metadata"]["executionProcessStatus"].as_str(),
        Some("running")
    );
    assert_eq!(
        output.after_raw[0].payload["metadata"]["outputBytes"].as_u64(),
        Some(7)
    );
    assert_eq!(
        output.after_raw[0].payload["delta"].as_str(),
        Some("running")
    );
}

#[test]
fn canonical_command_completion_emits_redacted_terminal_interaction_before_raw_item() {
    let mut mirror = CodingEventMirror::default();
    let payload = ThreadItemPayload::Command {
        command: "printf ready".to_string(),
        cwd: Some("/workspace".to_string()),
        output: Some("ready".to_string()),
        exit_code: Some(0),
    };
    let item = ThreadItem {
        session_id: SessionId::new("session-test"),
        thread_id: ThreadId::new("thread-test"),
        turn_id: TurnId::new("turn-test"),
        item_id: ItemId::new("exec-call"),
        sequence: 2,
        ordinal: 1,
        created_at_ms: 1,
        updated_at_ms: 2,
        completed_at_ms: Some(2),
        kind: payload.kind(),
        status: ItemStatus::Completed,
        payload,
        metadata: json!({
            "exec_command_call_id": "exec-call",
            "terminal_interaction": {
                "process_id": "process-7",
                "stdin": "sent 9 chars"
            }
        }),
    };

    let output = mirror.process_event(&RuntimeAgentEvent::ItemCompleted { item });

    assert_eq!(output.before_raw.len(), 1);
    assert!(output.after_raw.is_empty());
    assert_eq!(output.before_raw[0].event_type, "command.interaction");
    assert_eq!(output.before_raw[0].payload["commandId"], "item_exec-call");
    assert_eq!(output.before_raw[0].payload["processId"], "process-7");
    assert_eq!(output.before_raw[0].payload["stdin"], "sent 9 chars");
    assert!(!output.before_raw[0]
        .payload
        .to_string()
        .contains("continue"));
}

#[test]
fn shell_tool_metadata_only_delta_updates_process_lifecycle_without_consuming_output() {
    let mut mirror = CodingEventMirror::default();

    let _ = mirror.process_event(&tool_started(
        "Bash",
        "tool-process-start",
        Some(json!({ "command": "sleep 1 && echo done" })),
    ));
    let process_update = mirror.process_event(&RuntimeAgentEvent::ToolOutputDelta {
        tool_id: "tool-process-start".to_string(),
        delta: String::new(),
        output_kind: Some("process".to_string()),
        metadata: Some(HashMap::from([
            ("processId".to_string(), json!("process-tool-process-start")),
            ("executionProcessStatus".to_string(), json!("running")),
            (
                "executionProcessControlStatus".to_string(),
                json!("registered"),
            ),
            ("executionSurface".to_string(), json!("live_process")),
            ("stdinWritable".to_string(), json!(true)),
        ])),
    });
    let ended = mirror.process_event(&tool_completed(
        "tool-process-start",
        success_result(
            "done",
            HashMap::from([
                ("exit_code".to_string(), json!(0)),
                ("command".to_string(), json!("sleep 1 && echo done")),
            ]),
        ),
    ));

    assert_eq!(process_update.after_raw.len(), 1);
    assert_eq!(process_update.after_raw[0].event_type, "command.output");
    assert_eq!(
        process_update.after_raw[0].payload["metadata"]["processId"].as_str(),
        Some("process-tool-process-start")
    );
    assert_eq!(
        process_update.after_raw[0].payload["metadata"]["executionProcessStatus"].as_str(),
        Some("running")
    );
    assert_eq!(
        process_update.after_raw[0].payload["metadata"]["executionProcessControlStatus"].as_str(),
        Some("registered")
    );
    assert_eq!(
        process_update.after_raw[0].payload["metadata"]["stdinWritable"].as_bool(),
        Some(true)
    );
    assert!(process_update.after_raw[0].payload.get("preview").is_none());
    assert!(process_update.after_raw[0].payload.get("delta").is_none());
    assert_eq!(ended.after_raw[0].event_type, "command.output");
    assert_eq!(ended.after_raw[0].payload["delta"].as_str(), Some("done"));
    assert_eq!(ended.after_raw[0].payload["preview"].as_str(), Some("done"));
}

#[test]
fn shell_tool_result_emits_output_when_stream_delta_was_absent() {
    let mut mirror = CodingEventMirror::default();
    let _ = mirror.process_event(&tool_started(
        "PowerShellTool",
        "tool-2",
        Some(json!({ "command": "Write-Output ok" })),
    ));

    let ended = mirror.process_event(&tool_completed(
        "tool-2",
        success_result(
            "ok",
            HashMap::from([
                ("exit_code".to_string(), json!(0)),
                ("command".to_string(), json!("Write-Output ok")),
                ("execution_surface".to_string(), json!("embedded")),
                ("outputBytes".to_string(), json!(2)),
                ("outputOmittedBytes".to_string(), json!(0)),
                ("outputTruncated".to_string(), json!(false)),
                ("stdout_bytes".to_string(), json!(2)),
                ("stderr_bytes".to_string(), json!(0)),
            ]),
        ),
    ));

    assert_eq!(ended.after_raw[0].event_type, "command.output");
    assert_eq!(ended.after_raw[1].event_type, "command.exited");
    assert_eq!(ended.after_raw[0].payload["delta"].as_str(), Some("ok"));
    assert!(ended.after_raw[0].payload["outputRef"]
        .as_str()
        .is_some_and(|value| value.starts_with("output:command:")));
    assert_eq!(
        ended.after_raw[0].payload["processId"].as_str(),
        Some("process-tool-2")
    );
    assert_eq!(
        ended.after_raw[0].payload["executionProcessStatus"].as_str(),
        Some("exited")
    );
    assert_eq!(
        ended.after_raw[0].payload["executionSurface"].as_str(),
        Some("embedded")
    );
    assert_eq!(ended.after_raw[0].payload["outputBytes"].as_u64(), Some(2));
    assert_eq!(
        ended.after_raw[0].payload["metadata"]["stdoutBytes"].as_u64(),
        Some(2)
    );
    assert_eq!(
        ended.after_raw[1].payload["processId"].as_str(),
        Some("process-tool-2")
    );
    assert_eq!(
        ended.after_raw[1].payload["executionProcessStatus"].as_str(),
        Some("exited")
    );
    assert_eq!(
        ended.after_raw[1].payload["outputTruncated"].as_bool(),
        Some(false)
    );
}

#[test]
fn shell_apply_patch_command_emits_patch_lifecycle_after_command_exit() {
    let mut mirror = CodingEventMirror::default();
    let started = mirror.process_event(&tool_started(
        "Bash",
        "tool-patch-shell",
        Some(
            json!({
                "command": "apply_patch <<'PATCH'\n*** Begin Patch\n*** Add File: notes/live.md\n+hello\n*** End Patch\nPATCH"
            }),
        ),
    ));

    let ended = mirror.process_event(&tool_completed(
        "tool-patch-shell",
        success_result(
            "ok",
            HashMap::from([
                ("exit_code".to_string(), json!(0)),
                ("command".to_string(), json!("apply_patch <<'PATCH'")),
            ]),
        ),
    ));

    let event_types = started
        .after_raw
        .into_iter()
        .chain(ended.after_raw)
        .map(|event| event.event_type)
        .collect::<Vec<_>>();
    assert_eq!(
        event_types,
        vec![
            "patch.started",
            "command.started",
            "command.output",
            "command.exited",
            "patch.applied"
        ]
    );
}

#[test]
fn apply_patch_tool_failure_emits_patch_failed_with_category() {
    let mut mirror = CodingEventMirror::default();
    let started = mirror.process_event(&tool_started(
        "apply_patch",
        "tool-patch-failed",
        Some(
            json!({
                "patch": "*** Begin Patch\n*** Update File: missing.txt\n@@\n-old\n+new\n*** End Patch\n"
            }),
        ),
    ));

    let ended = mirror.process_event(&tool_completed(
        "tool-patch-failed",
        AgentToolResult {
            success: false,
            output: "target file not found".to_string(),
            error: None,
            structured_content: None,
            images: None,
            metadata: None,
        },
    ));

    assert_eq!(started.after_raw.len(), 1);
    assert_eq!(started.after_raw[0].event_type, "patch.started");
    assert_eq!(ended.after_raw.len(), 1);
    assert_eq!(ended.after_raw[0].event_type, "patch.failed");
    assert_eq!(
        ended.after_raw[0].payload["failureCategory"].as_str(),
        Some("missing_target")
    );
}

#[test]
fn declined_apply_patch_emits_declined_terminal_with_start_snapshot() {
    let mut mirror = CodingEventMirror::default();
    let started = mirror.process_event(&tool_started(
        "apply_patch",
        "tool-patch-declined",
        Some(json!({
            "patch": "*** Begin Patch\n*** Add File: declined.txt\n+never applied\n*** End Patch\n"
        })),
    ));
    let ended = mirror.process_event(&tool_completed(
        "tool-patch-declined",
        AgentToolResult {
            success: false,
            output: "用户拒绝工具执行".to_string(),
            error: None,
            structured_content: None,
            images: None,
            metadata: Some(HashMap::from([(
                "reasonCode".to_string(),
                json!("tool_approval_declined"),
            )])),
        },
    ));

    assert_eq!(started.after_raw[0].event_type, "patch.started");
    assert_eq!(
        started.after_raw[0].payload["changes"][0]["path"].as_str(),
        Some("declined.txt")
    );
    assert_eq!(ended.after_raw.len(), 1);
    assert_eq!(ended.after_raw[0].event_type, "patch.declined");
    assert_eq!(
        ended.after_raw[0].payload["status"].as_str(),
        Some("declined")
    );
    assert_eq!(
        ended.after_raw[0].payload["changes"][0]["path"].as_str(),
        Some("declined.txt")
    );
}

#[test]
fn apply_patch_tool_success_emits_one_batch_patch_lifecycle() {
    let mut mirror = CodingEventMirror::default();
    let _ = mirror.process_event(&tool_started(
        "apply_patch",
        "tool-patch-success",
        Some(
            json!({
                "patch": "*** Begin Patch\n*** Add File: a.txt\n+a\n*** Add File: b.txt\n+b\n*** End Patch\n"
            }),
        ),
    ));

    let events = mirror.process_event(&tool_completed(
        "tool-patch-success",
        success_result(
            "applied",
            HashMap::from([(
                "file_changes".to_string(),
                json!({
                    "changes": [
                        {
                            "kind": "add",
                            "path": "a.txt",
                            "content_size": 2,
                            "checkpointRef": "checkpoint:file:a",
                            "contentRef": "content:file:a",
                            "diffRef": "diff:file:a",
                            "diff": [
                                { "kind": "add", "value": "a" }
                            ]
                        },
                        {
                            "kind": "add",
                            "path": "b.txt",
                            "content_size": 2,
                            "checkpointRef": "checkpoint:file:b",
                            "contentRef": "content:file:b",
                            "diffRef": "diff:file:b"
                        }
                    ]
                }),
            )]),
        ),
    ));

    let event_types = events
        .after_raw
        .iter()
        .map(|event| event.event_type.as_str())
        .collect::<Vec<_>>();
    assert_eq!(event_types, vec!["patch.applied"]);
    let first_file_change = events.after_raw.first().expect("patch.applied event");
    let changes = first_file_change.payload["changes"]
        .as_array()
        .expect("changes");
    assert_eq!(changes.len(), 2);
    assert_eq!(changes[0]["path"].as_str(), Some("a.txt"));
    assert_eq!(changes[0]["kind"].as_str(), Some("add"));
    assert_eq!(changes[0]["diff"].as_str(), Some("+a"));
}

#[test]
fn write_tool_result_emits_file_changed_with_artifact_reference() {
    let mut mirror = CodingEventMirror::default();
    let _ = mirror.process_event(&tool_started(
        "Write",
        "tool-3",
        Some(json!({ "path": "src/App.tsx", "content": "export {}" })),
    ));

    let events = mirror.process_event(&tool_completed(
        "tool-3",
        success_result(
            "written",
            HashMap::from([("path".to_string(), json!("src/App.tsx"))]),
        ),
    ));

    assert_eq!(events.after_raw.len(), 1);
    assert_eq!(events.after_raw[0].event_type, "file.changed");
    assert_eq!(
        events.after_raw[0].payload["path"].as_str(),
        Some("src/App.tsx")
    );
    assert!(events.after_raw[0].payload["artifactId"].as_str().is_some());
}

#[test]
fn write_tool_result_preserves_artifact_checkpoint_and_diff_refs() {
    let mut mirror = CodingEventMirror::default();
    let _ = mirror.process_event(&tool_started(
        "Edit",
        "tool-edit-refs",
        Some(json!({ "path": "src/App.tsx" })),
    ));

    let events = mirror.process_event(&tool_completed(
        "tool-edit-refs",
        success_result(
            "updated",
            HashMap::from([
                ("path".to_string(), json!("src/App.tsx")),
                ("artifactId".to_string(), json!("artifact_src_app_after")),
                (
                    "artifactRefs".to_string(),
                    json!(["artifact_src_app_after", "artifact_src_app_before"]),
                ),
                (
                    "checkpointRef".to_string(),
                    json!("checkpoint_src_app_after"),
                ),
                ("contentRef".to_string(), json!("content://src-app-after")),
                ("diffRef".to_string(), json!("diff://src-app")),
                ("previewText".to_string(), json!("changed App component")),
            ]),
        ),
    ));

    let event = events.after_raw.first().expect("file.changed event");
    assert_eq!(event.event_type, "file.changed");
    assert_eq!(
        event.payload["artifactId"].as_str(),
        Some("artifact_src_app_after")
    );
    assert_eq!(
        event.payload["artifactRefs"]
            .as_array()
            .expect("artifact refs"),
        &vec![
            json!("artifact_src_app_after"),
            json!("artifact_src_app_before")
        ]
    );
    assert_eq!(
        event.payload["checkpointRef"].as_str(),
        Some("checkpoint_src_app_after")
    );
    assert_eq!(
        event.payload["contentRef"].as_str(),
        Some("content://src-app-after")
    );
    assert_eq!(event.payload["diffRef"].as_str(), Some("diff://src-app"));
    assert_eq!(
        event.payload["preview"].as_str(),
        Some("changed App component")
    );
}

#[test]
fn read_tool_result_emits_file_read_from_arguments() {
    let mut mirror = CodingEventMirror::default();
    let _ = mirror.process_event(&tool_started(
        "Read",
        "tool-read",
        Some(json!({ "path": "src/App.tsx", "start_line": 2, "end_line": 8 })),
    ));

    let events = mirror.process_event(&tool_completed(
        "tool-read",
        success_result(
            "2 | export {}",
            HashMap::from([("file_type".to_string(), json!("text"))]),
        ),
    ));

    assert_eq!(events.after_raw.len(), 1);
    assert_eq!(events.after_raw[0].event_type, "file.read");
    assert_eq!(
        events.after_raw[0].payload["path"].as_str(),
        Some("src/App.tsx")
    );
    assert_eq!(events.after_raw[0].payload["startLine"].as_u64(), Some(2));
    assert_eq!(events.after_raw[0].payload["endLine"].as_u64(), Some(8));
    assert!(events.after_raw[0].payload["outputRef"]
        .as_str()
        .is_some_and(|value| value.starts_with("output:file:")));
}

#[test]
fn read_and_shell_tool_results_preserve_output_refs() {
    let mut mirror = CodingEventMirror::default();
    let _ = mirror.process_event(&tool_started(
        "Read",
        "tool-read-ref",
        Some(json!({ "path": "src/App.tsx" })),
    ));
    let read_events = mirror.process_event(&tool_completed(
        "tool-read-ref",
        success_result(
            "export {}",
            HashMap::from([
                ("path".to_string(), json!("src/App.tsx")),
                ("outputRef".to_string(), json!("output://file-read")),
                ("contentRef".to_string(), json!("content://file-read")),
                (
                    "refIds".to_string(),
                    json!(["output://file-read", "trace://read"]),
                ),
            ]),
        ),
    ));
    let read = read_events.after_raw.first().expect("file.read event");
    assert_eq!(
        read.payload["outputRef"].as_str(),
        Some("output://file-read")
    );
    assert_eq!(
        read.payload["contentRef"].as_str(),
        Some("content://file-read")
    );
    assert_eq!(
        read.payload["refIds"].as_array().expect("read refs"),
        &vec![
            json!("output://file-read"),
            json!("trace://read"),
            json!("content://file-read")
        ]
    );

    let _ = mirror.process_event(&tool_started(
        "Bash",
        "tool-shell-ref",
        Some(json!({ "command": "npm test" })),
    ));
    let shell_events = mirror.process_event(&tool_completed(
        "tool-shell-ref",
        success_result(
            "ok",
            HashMap::from([
                ("exit_code".to_string(), json!(0)),
                ("outputRef".to_string(), json!("output://npm-test")),
                (
                    "refIds".to_string(),
                    json!(["output://npm-test", "log://npm-test"]),
                ),
            ]),
        ),
    ));
    let output = shell_events
        .after_raw
        .iter()
        .find(|event| event.event_type == "command.output")
        .expect("command.output event");
    assert_eq!(
        output.payload["outputRef"].as_str(),
        Some("output://npm-test")
    );
    assert_eq!(
        output.payload["refIds"].as_array().expect("command refs"),
        &vec![json!("output://npm-test"), json!("log://npm-test")]
    );
}

#[test]
fn failed_edit_does_not_emit_file_changed() {
    let mut mirror = CodingEventMirror::default();
    let _ = mirror.process_event(&tool_started(
        "Edit",
        "tool-4",
        Some(json!({ "path": "src/App.tsx" })),
    ));

    let events = mirror.process_event(&tool_completed(
        "tool-4",
        AgentToolResult {
            success: false,
            output: "not found".to_string(),
            error: Some("not found".to_string()),
            structured_content: None,
            images: None,
            metadata: Some(HashMap::from([("path".to_string(), json!("src/App.tsx"))])),
        },
    ));

    assert!(events.after_raw.is_empty());
}

#[test]
fn failed_tool_result_emits_permission_denied_before_raw_terminal() {
    let mut mirror = CodingEventMirror::default();
    let _ = mirror.process_event(&tool_started(
        "Bash",
        "tool-denied",
        Some(json!({ "command": "rm -rf important" })),
    ));

    let events = mirror.process_event(&tool_completed(
        "tool-denied",
        AgentToolResult {
            success: false,
            output: String::new(),
            error: Some("policy denied this command".to_string()),
            structured_content: None,
            images: None,
            metadata: Some(HashMap::from([
                ("reasonCode".to_string(), json!("dangerous_command")),
                ("policyName".to_string(), json!("default-shell-policy")),
                ("policyProfile".to_string(), json!("workspace-write")),
                ("policyDecisionId".to_string(), json!("decision-123")),
                ("platform".to_string(), json!("macos")),
            ])),
        },
    ));

    assert_eq!(events.before_raw.len(), 1);
    assert_eq!(events.before_raw[0].event_type, "permission.denied");
    assert_eq!(
        events.before_raw[0].payload["reasonCode"].as_str(),
        Some("dangerous_command")
    );
    assert_eq!(
        events.before_raw[0].payload["policyName"].as_str(),
        Some("default-shell-policy")
    );
    assert_eq!(
        events.before_raw[0].payload["policyProfile"].as_str(),
        Some("workspace-write")
    );
    assert_eq!(
        events.before_raw[0].payload["policyDecisionId"].as_str(),
        Some("decision-123")
    );
    assert_eq!(
        events.before_raw[0].payload["platform"].as_str(),
        Some("macos")
    );
    assert_eq!(
        events.before_raw[0].payload["command"].as_str(),
        Some("rm -rf important")
    );
    assert_eq!(
        events.before_raw[0].payload["diagnostics"]["toolSurface"].as_str(),
        Some("runtime_tool")
    );
}

#[test]
fn failed_tool_result_uses_policy_metadata_command_when_start_arguments_are_missing() {
    let mut mirror = CodingEventMirror::default();
    let _ = mirror.process_event(&tool_started("Bash", "tool-denied-metadata", None));

    let events = mirror.process_event(&tool_completed(
        "tool-denied-metadata",
        AgentToolResult {
            success: false,
            output: String::new(),
            error: Some("Permission denied: policy denied this command".to_string()),
            structured_content: None,
            images: None,
            metadata: Some(HashMap::from([
                ("eventClass".to_string(), json!("permission.denied")),
                ("command".to_string(), json!("rm -rf /tmp/outside")),
                ("cwd".to_string(), json!("/workspace")),
                ("policyName".to_string(), json!("workspace_tool_execution")),
            ])),
        },
    ));

    assert_eq!(events.before_raw.len(), 1);
    assert_eq!(events.before_raw[0].event_type, "permission.denied");
    assert_eq!(
        events.before_raw[0].payload["command"].as_str(),
        Some("rm -rf /tmp/outside")
    );
    assert_eq!(
        events.before_raw[0].payload["cwd"].as_str(),
        Some("/workspace")
    );
    assert_eq!(
        events.before_raw[0].payload["diagnostics"]["command"].as_str(),
        Some("rm -rf /tmp/outside")
    );
}

#[test]
fn failed_tool_result_emits_sandbox_blocked_before_raw_terminal() {
    let mut mirror = CodingEventMirror::default();
    let _ = mirror.process_event(&tool_started(
        "Bash",
        "tool-sandbox",
        Some(json!({ "command": "curl https://example.com" })),
    ));

    let events = mirror.process_event(&tool_completed(
        "tool-sandbox",
        AgentToolResult {
            success: false,
            output: "sandbox blocked network access".to_string(),
            error: None,
            structured_content: None,
            images: None,
            metadata: Some(HashMap::from([
                (
                    "failureCategory".to_string(),
                    json!("sandbox_network_blocked"),
                ),
                ("sandboxPolicy".to_string(), json!("network-disabled")),
                ("policyProfile".to_string(), json!("read-only")),
                ("platform".to_string(), json!("windows")),
            ])),
        },
    ));

    assert_eq!(events.before_raw.len(), 1);
    assert_eq!(events.before_raw[0].event_type, "sandbox.blocked");
    assert_eq!(
        events.before_raw[0].payload["reasonCode"].as_str(),
        Some("sandbox_blocked")
    );
    assert_eq!(
        events.before_raw[0].payload["sandboxPolicy"].as_str(),
        Some("network-disabled")
    );
    assert_eq!(
        events.before_raw[0].payload["policyProfile"].as_str(),
        Some("read-only")
    );
    assert_eq!(
        events.before_raw[0].payload["platform"].as_str(),
        Some("windows")
    );
    assert_eq!(
        events.before_raw[0].payload["command"].as_str(),
        Some("curl https://example.com")
    );
}
