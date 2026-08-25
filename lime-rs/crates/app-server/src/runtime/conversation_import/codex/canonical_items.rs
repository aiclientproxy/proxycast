use super::events::{visible_tool_output_text, CodexRolloutEvent, CodexToolCall, CodexToolPhase};
use super::history_builder::tool_output_text;
use agent_protocol::{
    AgentInput, CollabAgentOperation, ItemId, ItemStatus, SessionId, ThreadId, ThreadItem,
    ThreadItemPayload, ToolArgument, ToolOutput, TurnId,
};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

pub(super) fn project_rollout_events_to_canonical(
    events: &[CodexRolloutEvent],
    session_id: &str,
    thread_id: &str,
    turn_id: &str,
) -> Vec<CodexRolloutEvent> {
    let mut lowered = Vec::with_capacity(events.len());
    let mut active_tools = BTreeMap::<String, CodexToolCall>::new();
    let has_exec_tool = events.iter().any(|event| {
        event
            .tool_call()
            .and_then(|tool| tool.name.as_deref())
            .is_some_and(is_exec_command_tool)
    });

    for (index, event) in events.iter().enumerate() {
        if has_exec_tool && event.event_type().starts_with("command.") {
            continue;
        }
        if let Some(message_events) =
            lowered_message_item_events(event, session_id, thread_id, turn_id)
        {
            lowered.extend(message_events);
            continue;
        }
        if let Some(event) = canonical_single_item_with_source_ordinal(event) {
            lowered.push(event);
            continue;
        }
        let Some(tool) = event.tool_call() else {
            lowered.push(event.clone());
            continue;
        };
        let ordinal = source_event_ordinal(tool).unwrap_or(index as u64 + 1);
        let call_id = tool
            .call_id
            .as_deref()
            .expect("normalized imported tool draft must have call id");
        if is_represented_by_specialized_item(tool, events) {
            continue;
        }
        if tool.phase == CodexToolPhase::Started {
            active_tools.insert(call_id.to_string(), tool.clone());
            lowered.push(lowered_tool_item_event(
                "item.started",
                session_id,
                thread_id,
                turn_id,
                call_id,
                ordinal,
                tool,
                None,
            ));
            continue;
        }

        let active_start = active_tools.remove(call_id);
        let had_start = active_start.is_some();
        let start = active_start.unwrap_or_else(|| tool.clone());
        if !had_start {
            let start_ordinal = source_event_ordinal(&start).unwrap_or(ordinal);
            lowered.push(lowered_tool_item_event(
                "item.started",
                session_id,
                thread_id,
                turn_id,
                call_id,
                start_ordinal,
                &start,
                None,
            ));
        }
        lowered.push(lowered_tool_item_event(
            "item.completed",
            session_id,
            thread_id,
            turn_id,
            call_id,
            ordinal,
            &start,
            Some(tool),
        ));
    }

    lowered
}

fn lowered_message_item_events(
    event: &CodexRolloutEvent,
    session_id: &str,
    thread_id: &str,
    turn_id: &str,
) -> Option<Vec<CodexRolloutEvent>> {
    let event_type = event.event_type();
    if event_type != "import.message" && event_type != "message.delta" {
        return None;
    }
    let source = event.payload()?.as_object()?;
    if event_type == "message.delta"
        && source.get("sourceClient").and_then(Value::as_str) != Some("codex")
    {
        return None;
    }
    let role = source
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or("assistant");
    let item_role = if role == "user" { "user" } else { "agent" };
    let source_provenance = source.get("sourceProvenance").cloned()?;
    let ordinal = source
        .get("ordinal")
        .and_then(Value::as_u64)
        .or_else(|| source_event_ordinal_from_provenance(&source_provenance))?;
    let item_id = source
        .get("itemId")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .or_else(|| {
            source_provenance
                .get("sourceCallId")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .map(str::to_string)
        })
        .unwrap_or_else(|| format!("imported-{item_role}_{ordinal}"));
    let text = source
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let attachments = source
        .get("attachments")
        .cloned()
        .unwrap_or_else(|| json!([]));
    let input = source
        .get("input")
        .and_then(|value| serde_json::from_value::<Vec<AgentInput>>(value.clone()).ok())
        .unwrap_or_else(|| vec![AgentInput::text(text.clone())]);
    let phase = source.get("phase").cloned().unwrap_or(Value::Null);
    let client_id = source.get("clientId").cloned().unwrap_or(Value::Null);
    let source_event_type = source_provenance
        .get("sourcePayloadType")
        .cloned()
        .unwrap_or(Value::Null);
    let source_call_id = source_provenance
        .get("sourceCallId")
        .cloned()
        .unwrap_or(Value::Null);
    let metadata = json!({
        "imported": true,
        "source_client": "codex",
        "source_event_seq": ordinal,
        "source_event_type": source_event_type,
        "source_call_id": source_call_id,
        "source_provenance": source_provenance,
    });
    let item_payload = if role == "user" {
        json!({
            "type": "userMessage",
            "content": input,
            "clientId": client_id,
        })
    } else {
        json!({
            "type": "agentMessage",
            "text": text,
            "phase": phase,
        })
    };
    let started_item = canonical_message_item(
        session_id,
        thread_id,
        turn_id,
        &item_id,
        ordinal,
        item_role,
        "inProgress",
        item_payload.clone(),
        metadata.clone(),
        None,
    );
    let completed_item = canonical_message_item(
        session_id,
        thread_id,
        turn_id,
        &item_id,
        ordinal,
        item_role,
        "completed",
        item_payload,
        metadata,
        Some(0),
    );
    let outer_metadata = json!({
        "imported": true,
        "sourceClient": "codex",
        "sourceEventSeq": ordinal,
        "sourceProvenance": source_provenance,
    });
    let presentation = if role == "user" {
        CodexRolloutEvent::new(
            "message.created",
            merge_json_objects(
                json!({
                    "itemId": item_id,
                    "ordinal": ordinal,
                    "role": "user",
                    "visibility": "user_visible",
                    "input": input,
                    "content": {"kind": "inline_text", "text": text},
                    "attachments": attachments,
                }),
                outer_metadata.clone(),
            ),
        )
    } else {
        CodexRolloutEvent::new(
            "message.delta",
            merge_json_objects(
                json!({
                    "itemId": item_id,
                    "ordinal": ordinal,
                    "role": "assistant",
                    "text": text,
                    "phase": phase,
                }),
                outer_metadata.clone(),
            ),
        )
    };

    Some(vec![
        CodexRolloutEvent::new(
            "item.started",
            merge_json_objects(json!({"item": started_item}), outer_metadata.clone()),
        ),
        presentation,
        CodexRolloutEvent::new(
            "item.completed",
            merge_json_objects(json!({"item": completed_item}), outer_metadata),
        ),
    ])
}

#[allow(clippy::too_many_arguments)]
fn canonical_message_item(
    session_id: &str,
    thread_id: &str,
    turn_id: &str,
    item_id: &str,
    ordinal: u64,
    item_role: &str,
    status: &str,
    payload: Value,
    metadata: Value,
    completed_at_ms: Option<i64>,
) -> Value {
    json!({
        "sessionId": session_id,
        "threadId": thread_id,
        "turnId": turn_id,
        "itemId": item_id,
        "sequence": 0,
        "ordinal": ordinal,
        "createdAtMs": 0,
        "updatedAtMs": 0,
        "completedAtMs": completed_at_ms,
        "kind": if item_role == "user" { "userMessage" } else { "agentMessage" },
        "status": status,
        "payload": payload,
        "metadata": metadata,
    })
}

fn canonical_single_item_with_source_ordinal(
    event: &CodexRolloutEvent,
) -> Option<CodexRolloutEvent> {
    let payload = event.payload()?;
    let source_provenance = payload.get("sourceProvenance")?;
    let uses_source_ordinal = event.event_type() == "reasoning.completed"
        || (event.event_type() == "plan.final" && payload.get("itemId").is_some());
    if !uses_source_ordinal {
        return None;
    }
    let ordinal = source_event_ordinal_from_provenance(source_provenance)?;
    let mut payload = payload.clone();
    let payload = payload.as_object_mut()?;
    payload
        .entry("ordinal".to_string())
        .or_insert_with(|| json!(ordinal));
    let metadata = payload
        .entry("metadata".to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    if let Some(metadata) = metadata.as_object_mut() {
        metadata
            .entry("source_client".to_string())
            .or_insert_with(|| json!("codex"));
        metadata
            .entry("source_event_seq".to_string())
            .or_insert_with(|| json!(ordinal));
        metadata
            .entry("source_provenance".to_string())
            .or_insert_with(|| source_provenance.clone());
    }
    Some(CodexRolloutEvent::new(
        event.event_type(),
        Value::Object(payload.clone()),
    ))
}

fn source_event_ordinal_from_provenance(provenance: &Value) -> Option<u64> {
    provenance
        .get("sourceEventSeq")
        .and_then(|value| value.as_u64().or_else(|| value.as_str()?.parse().ok()))
}

fn merge_json_objects(mut primary: Value, extra: Value) -> Value {
    if let (Some(primary), Some(extra)) = (primary.as_object_mut(), extra.as_object()) {
        primary.extend(extra.clone());
    }
    primary
}

fn lowered_tool_item_event(
    event_type: &'static str,
    session_id: &str,
    thread_id: &str,
    turn_id: &str,
    call_id: &str,
    ordinal: u64,
    start: &CodexToolCall,
    terminal: Option<&CodexToolCall>,
) -> CodexRolloutEvent {
    let source = terminal.unwrap_or(start);
    let failed = terminal.is_some_and(|tool| tool.phase == CodexToolPhase::Failed);
    let tool_name = start
        .name
        .clone()
        .or_else(|| source.name.clone())
        .unwrap_or_else(|| "unknown_tool".to_string());
    let raw_arguments = start.arguments.as_ref().or(source.arguments.as_ref());
    let arguments = typed_arguments(raw_arguments);
    let metadata = tool_metadata(start, terminal);
    let output = terminal.and_then(|tool| typed_tool_output(tool, failed));
    let status = if failed {
        ItemStatus::Failed
    } else if terminal.is_some() {
        ItemStatus::Completed
    } else {
        ItemStatus::InProgress
    };
    let (item_id, payload) = canonical_tool_payload(
        call_id,
        &tool_name,
        raw_arguments,
        arguments,
        output,
        terminal,
    );
    let mut item = ThreadItem::new(
        SessionId::new(session_id),
        ThreadId::new(thread_id),
        TurnId::new(turn_id),
        0,
        ordinal,
        payload,
    );
    item.item_id = ItemId::from_legacy(item_id);
    item.status = status;
    item.completed_at_ms = terminal.map(|_| 0);
    item.metadata = metadata;
    let item = serde_json::to_value(item).expect("canonical imported tool item must serialize");
    let mut payload = Map::new();
    payload.insert("item".to_string(), item);
    copy_outer_tool_metadata(&mut payload, start, terminal);
    CodexRolloutEvent::new(event_type, Value::Object(payload))
}

fn typed_arguments(arguments: Option<&Value>) -> Vec<ToolArgument> {
    let Some(arguments) = arguments else {
        return Vec::new();
    };
    match arguments {
        Value::Object(object) => object
            .iter()
            .map(|(name, value)| ToolArgument {
                name: name.clone(),
                value: value_to_argument_string(value),
            })
            .collect(),
        Value::Array(values) => values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                if let (Some(name), Some(value)) = (
                    value.get("name").and_then(Value::as_str),
                    value.get("value"),
                ) {
                    ToolArgument {
                        name: name.to_string(),
                        value: value_to_argument_string(value),
                    }
                } else {
                    ToolArgument {
                        name: index.to_string(),
                        value: value_to_argument_string(value),
                    }
                }
            })
            .collect(),
        value => vec![ToolArgument {
            name: "value".to_string(),
            value: value_to_argument_string(value),
        }],
    }
}

fn canonical_tool_payload(
    call_id: &str,
    tool_name: &str,
    raw_arguments: Option<&Value>,
    arguments: Vec<ToolArgument>,
    output: Option<ToolOutput>,
    terminal: Option<&CodexToolCall>,
) -> (String, ThreadItemPayload) {
    if is_exec_command_tool(tool_name) {
        let command = argument_string(raw_arguments, &["cmd", "command"])
            .unwrap_or_else(|| tool_name.to_string());
        let cwd = argument_string(raw_arguments, &["workdir", "cwd"]);
        let output_text = output.as_ref().and_then(|value| value.text.clone());
        let exit_code = terminal.and_then(command_exit_code);
        return (
            format!("codex-command-{call_id}"),
            ThreadItemPayload::Command {
                command,
                cwd,
                output: output_text,
                exit_code,
            },
        );
    }

    if is_mcp_tool(tool_name, terminal) {
        let (server_name, mcp_tool_name) = mcp_identity(tool_name, terminal);
        return (
            format!("codex-mcp-{call_id}"),
            ThreadItemPayload::McpToolCall {
                call_id: call_id.to_string(),
                server_name,
                tool_name: mcp_tool_name,
                app_context: None,
                mcp_app_resource_uri: None,
                plugin_id: None,
                arguments,
                output,
            },
        );
    }

    if let Some(operation) = collab_operation(tool_name) {
        return (
            format!("codex-collab-{call_id}"),
            ThreadItemPayload::CollabAgentToolCall {
                call_id: call_id.to_string(),
                operation,
                target_thread_id: argument_string(
                    raw_arguments,
                    &["target_thread_id", "thread_id", "agent_id"],
                )
                .map(ThreadId::new),
                message: argument_string(raw_arguments, &["message", "prompt"]),
                output,
                agent_states: Default::default(),
            },
        );
    }

    (
        format!("codex-tool-{call_id}"),
        ThreadItemPayload::Tool {
            call_id: call_id.to_string(),
            name: tool_name.to_string(),
            arguments,
            output,
        },
    )
}

fn argument_string(arguments: Option<&Value>, keys: &[&str]) -> Option<String> {
    let arguments = arguments?.as_object()?;
    keys.iter().find_map(|key| {
        arguments.get(*key).and_then(|value| {
            value
                .as_str()
                .map(str::to_string)
                .or_else(|| (!value.is_null()).then(|| value.to_string()))
        })
    })
}

fn command_exit_code(tool: &CodexToolCall) -> Option<i32> {
    let output = tool_output_text(tool)?;
    for line in output.lines() {
        for marker in ["Process exited with code ", "exit code ", "Exit code: "] {
            if let Some(value) = line.split_once(marker).map(|(_, value)| value.trim()) {
                if let Ok(code) = value
                    .trim_end_matches(|character: char| !character.is_ascii_digit())
                    .parse()
                {
                    return Some(code);
                }
            }
        }
    }
    Some(if tool.phase == CodexToolPhase::Failed {
        1
    } else {
        0
    })
}

fn is_exec_command_tool(tool_name: &str) -> bool {
    matches!(
        normalized_tool_name(tool_name).as_str(),
        "execcommand" | "executecommand" | "shell" | "bash" | "powershell"
    )
}

fn is_mcp_tool(tool_name: &str, tool: Option<&CodexToolCall>) -> bool {
    tool_name.trim().to_ascii_lowercase().starts_with("mcp__")
        || tool.is_some_and(|tool| {
            tool.source
                .source_event_type
                .as_deref()
                .is_some_and(|value| value.starts_with("mcp_tool_call"))
        })
}

fn mcp_identity(tool_name: &str, tool: Option<&CodexToolCall>) -> (String, String) {
    let mut parts = tool_name.split("__");
    if parts.next() == Some("mcp") {
        let server = parts.next().unwrap_or("unknown").to_string();
        let name = parts.collect::<Vec<_>>().join("__");
        return (
            server,
            if name.is_empty() {
                "tool".to_string()
            } else {
                name
            },
        );
    }
    let metadata = tool.map(|tool| tool.source.to_value());
    let server = metadata
        .as_ref()
        .and_then(|value| value.get("serverName").or_else(|| value.get("server_name")))
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    (server, tool_name.to_string())
}

fn collab_operation(tool_name: &str) -> Option<CollabAgentOperation> {
    match normalized_tool_name(tool_name).as_str() {
        "waitagent" => Some(CollabAgentOperation::Wait),
        _ => None,
    }
}

fn normalized_tool_name(tool_name: &str) -> String {
    tool_name
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn is_represented_by_specialized_item(tool: &CodexToolCall, events: &[CodexRolloutEvent]) -> bool {
    let Some(tool_name) = tool.name.as_deref() else {
        return false;
    };
    let normalized = normalized_tool_name(tool_name);
    if normalized == "updateplan" {
        return events
            .iter()
            .any(|event| event.event_type() == "plan.final");
    }
    if normalized == "applypatch" {
        return events
            .iter()
            .any(|event| event.event_type().starts_with("patch."));
    }
    false
}

fn value_to_argument_string(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| value.to_string())
}

fn tool_output(tool: &CodexToolCall, failed: bool) -> Value {
    let raw_output = tool.output.clone();
    let output_object = raw_output.as_ref().and_then(Value::as_object);
    let text = raw_output
        .as_ref()
        .and_then(visible_tool_output_text)
        .or_else(|| tool.source.output_preview.clone());
    let structured_content = tool.source.structured_content.clone().or_else(|| {
        output_object
            .and_then(|object| object.get("structuredContent"))
            .cloned()
    });
    let error = tool.source.error.clone().or_else(|| {
        output_object.and_then(|object| {
            ["error", "message"]
                .iter()
                .find_map(|key| object.get(*key).and_then(Value::as_str))
                .map(str::to_string)
        })
    });
    let duration_ms = tool.source.duration_ms;
    let truncated = tool.source.truncated;
    let output_ref = tool.source.output_ref.clone();
    let mut output = Map::new();
    if let Some(text) = text {
        output.insert("text".to_string(), Value::String(text));
    }
    if !tool.source.imported {
        if let Some(structured_content) = structured_content {
            output.insert("structuredContent".to_string(), structured_content);
        } else if let Some(raw_output) = raw_output.filter(|value| !value.is_string()) {
            output.insert("structuredContent".to_string(), raw_output);
        }
    }
    if let Some(error) = error {
        output.insert("error".to_string(), Value::String(error));
    }
    if let Some(duration_ms) = duration_ms {
        output.insert("durationMs".to_string(), json!(duration_ms));
    }
    if truncated {
        output.insert("truncated".to_string(), Value::Bool(true));
    }
    if let Some(output_ref) = output_ref {
        output.insert("outputRef".to_string(), Value::String(output_ref));
    }
    if failed && !output.contains_key("error") {
        if let Some(failure) = tool.source.failure_category.clone() {
            output.insert("error".to_string(), Value::String(failure));
        }
    }
    Value::Object(output)
}

fn typed_tool_output(tool: &CodexToolCall, failed: bool) -> Option<ToolOutput> {
    let output = tool_output(tool, failed);
    serde_json::from_value(output).ok()
}

fn tool_metadata(start: &CodexToolCall, terminal: Option<&CodexToolCall>) -> Value {
    let mut metadata = Map::new();
    merge_tool_metadata(&mut metadata, start);
    if let Some(terminal) = terminal {
        if let Some(provenance) = start.source.source_provenance.as_ref() {
            metadata.insert("start_source_provenance".to_string(), provenance.clone());
        }
        merge_tool_metadata(&mut metadata, terminal);
        if let Some(provenance) = terminal.source.source_provenance.as_ref() {
            metadata.insert("terminal_source_provenance".to_string(), provenance.clone());
        }
    }
    Value::Object(metadata)
}

fn merge_tool_metadata(metadata: &mut Map<String, Value>, tool: &CodexToolCall) {
    insert_optional_string(metadata, "source_call_id", tool.call_id.as_ref());
    insert_true(metadata, "imported", tool.source.imported);
    insert_optional_string(
        metadata,
        "source_client",
        tool.source.source_client.as_ref(),
    );
    insert_optional_value(
        metadata,
        "source_provenance",
        tool.source.source_provenance.as_ref(),
    );
    insert_optional_string(
        metadata,
        "source_event_type",
        tool.source.source_event_type.as_ref(),
    );
    if let Some(sequence) = source_event_ordinal(tool) {
        metadata.insert("source_event_seq".to_string(), json!(sequence));
    }
    insert_true(metadata, "imported_synthetic", tool.source.synthetic);
    insert_true(metadata, "imported_incomplete", tool.source.incomplete);
    insert_true(metadata, "imported_synthetic_id", tool.source.synthetic_id);
    insert_optional_string(
        metadata,
        "failure_category",
        tool.source.failure_category.as_ref(),
    );
    insert_optional_value(metadata, "action", tool.source.action.as_ref());
    insert_optional_string(metadata, "query", tool.source.query.as_ref());
    if let Some(success) = tool.source.success {
        metadata.insert("success".to_string(), Value::Bool(success));
    }
    if !tool.source.ref_ids.is_empty() {
        metadata.insert("ref_ids".to_string(), json!(tool.source.ref_ids));
    }
    if let Some(output_bytes) = tool.source.output_bytes {
        metadata.insert("output_bytes".to_string(), json!(output_bytes));
    }
}

fn copy_outer_tool_metadata(
    payload: &mut Map<String, Value>,
    start: &CodexToolCall,
    terminal: Option<&CodexToolCall>,
) {
    let source = terminal.unwrap_or(start);
    insert_true(payload, "imported", source.source.imported);
    insert_optional_string(
        payload,
        "sourceClient",
        source.source.source_client.as_ref(),
    );
    insert_optional_value(
        payload,
        "sourceProvenance",
        source.source.source_provenance.as_ref(),
    );
    insert_optional_string(
        payload,
        "sourceEventType",
        source.source.source_event_type.as_ref(),
    );
    if let Some(sequence) = source_event_ordinal(source) {
        payload.insert("sourceEventSeq".to_string(), json!(sequence));
    }
    insert_true(payload, "importedSynthetic", source.source.synthetic);
    insert_true(payload, "importedIncomplete", source.source.incomplete);
    insert_true(payload, "importedSyntheticId", source.source.synthetic_id);
    insert_optional_string(
        payload,
        "failureCategory",
        source.source.failure_category.as_ref(),
    );
    insert_optional_value(payload, "action", source.source.action.as_ref());
    insert_optional_string(payload, "query", source.source.query.as_ref());
    if let Some(success) = source.source.success {
        payload.insert("success".to_string(), Value::Bool(success));
    }
    if !source.source.ref_ids.is_empty() {
        payload.insert("refIds".to_string(), json!(source.source.ref_ids));
    }
    if let Some(output_bytes) = source.source.output_bytes {
        payload.insert("outputBytes".to_string(), json!(output_bytes));
    }
    if let Some(start_provenance) = start.source.source_provenance.as_ref() {
        payload.insert(
            "startSourceProvenance".to_string(),
            start_provenance.clone(),
        );
    }
}

fn source_event_ordinal(tool: &CodexToolCall) -> Option<u64> {
    tool.source
        .source_provenance
        .as_ref()
        .and_then(|provenance| provenance.get("sourceEventSeq"))
        .and_then(|value| value.as_u64().or_else(|| value.as_str()?.parse().ok()))
}

fn insert_true(target: &mut Map<String, Value>, key: &str, value: bool) {
    if value {
        target.insert(key.to_string(), Value::Bool(true));
    }
}

fn insert_optional_string(target: &mut Map<String, Value>, key: &str, value: Option<&String>) {
    if let Some(value) = value {
        target.insert(key.to_string(), Value::String(value.clone()));
    }
}

fn insert_optional_value(target: &mut Map<String, Value>, key: &str, value: Option<&Value>) {
    if let Some(value) = value {
        target.insert(key.to_string(), value.clone());
    }
}
