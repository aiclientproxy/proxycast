use super::canonical_items::project_rollout_events_to_canonical;
use super::events::{
    visible_tool_output_text, CodexRolloutEvent, CodexToolCall, CodexToolPhase, CodexToolSource,
};
use crate::RuntimeEvent;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

#[cfg(test)]
#[path = "history_builder_tests.rs"]
mod tests;

pub(super) struct CodexHistoryBuilder {
    active_tools: BTreeMap<String, CodexToolCall>,
    completed_tools: BTreeSet<String>,
    active_patches: BTreeSet<String>,
    active_actions: BTreeSet<String>,
    active_commands: BTreeSet<String>,
    synthetic_tool_call_sequence: usize,
    pending_terminal_event: Option<CodexRolloutEvent>,
    saw_terminal_event: bool,
}

impl CodexHistoryBuilder {
    pub(super) fn new() -> Self {
        Self {
            active_tools: BTreeMap::new(),
            completed_tools: BTreeSet::new(),
            active_patches: BTreeSet::new(),
            active_actions: BTreeSet::new(),
            active_commands: BTreeSet::new(),
            synthetic_tool_call_sequence: 0,
            pending_terminal_event: None,
            saw_terminal_event: false,
        }
    }

    pub(super) fn push(&mut self, event: CodexRolloutEvent) -> Vec<CodexRolloutEvent> {
        if let Some(tool) = event.tool_call().cloned() {
            return self.push_tool(tool);
        }
        let mut normalized = Vec::new();
        match event.event_type() {
            "patch.started" => {
                if let Some(patch_id) = event
                    .payload()
                    .and_then(|payload| string_payload(payload, &["patchId", "patch_id"]))
                {
                    self.active_patches.insert(patch_id);
                }
                normalized.push(event);
            }
            "patch.applied" | "patch.failed" => {
                let patch_id = event
                    .payload()
                    .and_then(|payload| string_payload(payload, &["patchId", "patch_id"]));
                if let Some(patch_id) = patch_id.as_deref() {
                    if !self.active_patches.remove(patch_id) {
                        normalized.push(synthesized_patch_start(patch_id, &event));
                    }
                }
                normalized.push(event);
            }
            "command.started" => {
                if let Some(command_id) = event
                    .payload()
                    .and_then(|payload| string_payload(payload, &["commandId"]))
                {
                    self.active_commands.insert(command_id);
                }
                normalized.push(event);
            }
            "command.output" => normalized.push(event),
            "command.exited" => {
                if let Some(command_id) = event
                    .payload()
                    .and_then(|payload| string_payload(payload, &["commandId"]))
                {
                    self.active_commands.remove(&command_id);
                }
                normalized.push(event);
            }
            "action.required" => {
                let action_id = event.payload().and_then(|payload| {
                    string_payload(payload, &["actionId", "action_id", "requestId"])
                });
                if let Some(action_id) = action_id.as_deref() {
                    self.active_actions.insert(action_id.to_string());
                }
                normalized.push(event);
                if let Some(action_id) = action_id {
                    normalized.push(historical_action_resolved(&action_id));
                    self.active_actions.remove(&action_id);
                }
            }
            "action.resolved" | "action.cancelled" | "action.canceled" | "action.expired" => {
                if let Some(action_id) = event.payload().and_then(|payload| {
                    string_payload(payload, &["actionId", "action_id", "requestId"])
                }) {
                    self.active_actions.remove(&action_id);
                }
                normalized.push(event);
            }
            "turn.completed" | "turn.failed" | "turn.canceled" => {
                self.saw_terminal_event = true;
                if self.pending_terminal_event.is_none()
                    || event.event_type() == "turn.failed"
                    || event.event_type() == "turn.canceled"
                {
                    self.pending_terminal_event = Some(event);
                }
            }
            _ => normalized.push(event),
        }
        normalized
    }

    fn push_tool(&mut self, mut tool: CodexToolCall) -> Vec<CodexRolloutEvent> {
        let call_id = self.ensure_imported_tool_call_id(&mut tool);
        let mut normalized = Vec::new();
        match tool.phase {
            CodexToolPhase::Started => {
                if self.completed_tools.contains(&call_id) {
                    return normalized;
                }
                if let Some(active) = self.active_tools.get_mut(&call_id) {
                    if active.name.is_none() {
                        active.name = tool.name;
                    }
                    if active.arguments.is_none() {
                        active.arguments = tool.arguments;
                    }
                    return normalized;
                }
                self.active_tools.insert(call_id, tool.clone());
                normalized.push(CodexRolloutEvent::Tool(tool));
            }
            CodexToolPhase::Completed | CodexToolPhase::Failed => {
                if !self.completed_tools.insert(call_id.clone()) {
                    return normalized;
                }
                if let Some(start) = self.active_tools.remove(&call_id) {
                    if tool.name.is_none() {
                        tool.name = start.name.clone();
                    }
                    if tool.arguments.is_none() {
                        tool.arguments = start.arguments.clone();
                    }
                } else {
                    normalized.push(CodexRolloutEvent::Tool(synthesized_tool_start(
                        &call_id, &tool,
                    )));
                }
                if self.active_commands.remove(&call_id) {
                    normalized.push(synthesized_command_output(&call_id, &tool));
                    normalized.push(synthesized_command_exit(&call_id, &tool));
                }
                normalized.push(CodexRolloutEvent::Tool(tool.clone()));
                if let Some(file_event) = imported_file_changed_event(&tool) {
                    normalized.push(file_event);
                }
            }
        }
        normalized
    }

    fn ensure_imported_tool_call_id(&mut self, tool: &mut CodexToolCall) -> String {
        if let Some(call_id) = tool
            .call_id
            .as_deref()
            .filter(|value| !value.trim().is_empty())
        {
            return call_id.to_string();
        }
        self.synthetic_tool_call_sequence += 1;
        let source_sequence = tool
            .source
            .source_provenance
            .as_ref()
            .and_then(|provenance| provenance.get("sourceEventSeq"))
            .and_then(|value| value.as_u64().or_else(|| value.as_str()?.parse().ok()));
        let call_id = source_sequence
            .map(|sequence| format!("imported-tool-{sequence}"))
            .unwrap_or_else(|| {
                format!("imported-tool-local-{}", self.synthetic_tool_call_sequence)
            });
        tool.call_id = Some(call_id.clone());
        tool.source.synthetic_id = true;
        call_id
    }

    pub(super) fn finish(&mut self) -> Vec<CodexRolloutEvent> {
        let mut normalized = Vec::new();
        close_active_imported_lifecycles(
            &mut normalized,
            &mut self.active_actions,
            &mut self.active_commands,
            &mut self.active_patches,
            &mut self.active_tools,
        );
        if let Some(event) = self.pending_terminal_event.take() {
            normalized.push(event);
        }
        normalized
    }

    pub(super) fn has_terminal_event(&self) -> bool {
        self.saw_terminal_event
    }
}

fn imported_file_changed_event(tool: &CodexToolCall) -> Option<CodexRolloutEvent> {
    if tool.name.as_deref()?.trim() != "read_file" {
        return None;
    }
    let path = tool
        .arguments
        .as_ref()
        .and_then(|arguments| string_value(arguments, &["path", "filePath", "file_path"]))?;
    let content = tool_output_text(tool)?;
    let call_id = tool.call_id.as_deref().unwrap_or("read_file");
    Some(CodexRolloutEvent::new(
        "file.changed",
        json!({
            "path": path,
            "content": content,
            "artifactId": format!("codex-import-file-{call_id}"),
            "sourceClient": "codex",
            "eventClass": "file.read",
            "operation": "read",
            "toolName": "read_file",
            "sourceEventType": "function_call_output",
            "sourceProvenance": tool.source.source_provenance.clone(),
            "imported": true,
        }),
    ))
}

fn string_value(value: &Value, keys: &[&str]) -> Option<String> {
    let object = value.as_object()?;
    keys.iter().find_map(|key| {
        object.get(*key).and_then(|value| {
            value
                .as_str()
                .map(str::to_string)
                .or_else(|| (!value.is_null()).then(|| value.to_string()))
        })
    })
}

fn enrich_history_payload(mut payload: Value) -> Value {
    if let Value::Object(ref mut object) = payload {
        object
            .entry("imported".to_string())
            .or_insert_with(|| json!(true));
        object
            .entry("sourceClient".to_string())
            .or_insert_with(|| json!("codex"));
        object
            .entry("importVersion".to_string())
            .or_insert_with(|| json!(2));
    }
    payload
}

fn close_active_imported_lifecycles(
    normalized: &mut Vec<CodexRolloutEvent>,
    active_actions: &mut BTreeSet<String>,
    active_commands: &mut BTreeSet<String>,
    active_patches: &mut BTreeSet<String>,
    active_tools: &mut BTreeMap<String, CodexToolCall>,
) {
    let actions = std::mem::take(active_actions);
    for action_id in actions {
        normalized.push(historical_action_resolved(&action_id));
    }
    let commands = std::mem::take(active_commands);
    for command_id in commands {
        normalized.push(CodexRolloutEvent::new(
            "command.exited",
            json!({
                "commandId": command_id,
                "exitCode": 1,
                "status": "failed",
                "result": "imported_incomplete",
                "failureCategory": "incomplete_import",
                "sourceClient": "codex",
                "sourceEventType": "synthetic_command_exited",
                "importedSynthetic": true,
                "importedIncomplete": true,
            }),
        ));
    }
    let patches = std::mem::take(active_patches);
    for patch_id in patches {
        normalized.push(CodexRolloutEvent::new(
            "patch.failed",
            json!({
                "patchId": patch_id,
                "status": "failed",
                "success": false,
                "failureCategory": "incomplete_import",
                "sourceClient": "codex",
                "sourceEventType": "synthetic_patch_failed",
                "importedSynthetic": true,
                "importedIncomplete": true,
            }),
        ));
    }
    let tools = std::mem::take(active_tools);
    for (_, start) in tools {
        let mut terminal = start.clone();
        terminal.phase = CodexToolPhase::Failed;
        terminal.output = None;
        terminal.source.source_client = Some("codex".to_string());
        terminal.source.source_event_type = Some("synthetic_tool_result".to_string());
        terminal.source.success = Some(false);
        terminal.source.failure_category = Some("incomplete_import".to_string());
        terminal.source.error = Some("source tool lifecycle is incomplete".to_string());
        terminal.source.synthetic = true;
        terminal.source.incomplete = true;
        normalized.push(CodexRolloutEvent::Tool(terminal));
    }
}

fn synthesized_tool_start(tool_call_id: &str, terminal: &CodexToolCall) -> CodexToolCall {
    CodexToolCall {
        phase: CodexToolPhase::Started,
        call_id: Some(tool_call_id.to_string()),
        name: terminal.name.clone(),
        arguments: terminal.arguments.clone(),
        output: None,
        source: CodexToolSource {
            source_client: Some("codex".to_string()),
            source_event_type: Some("synthetic_tool_started".to_string()),
            synthetic: true,
            action: terminal.source.action.clone(),
            query: terminal.source.query.clone(),
            ..CodexToolSource::default()
        },
    }
}

fn synthesized_patch_start(patch_id: &str, terminal: &CodexRolloutEvent) -> CodexRolloutEvent {
    let payload = terminal
        .payload()
        .expect("patch terminal must be a runtime event");
    CodexRolloutEvent::new(
        "patch.started",
        json!({
            "patchId": patch_id,
            "toolCallId": string_payload(payload, &["toolCallId", "tool_call_id"]),
            "paths": payload.get("paths").cloned(),
            "changedFiles": payload.get("changedFiles").cloned(),
            "sourceClient": "codex",
            "sourceEventType": "synthetic_patch_started",
            "importedSynthetic": true,
        }),
    )
}

fn synthesized_command_output(command_id: &str, terminal: &CodexToolCall) -> CodexRolloutEvent {
    let output = tool_output_text(terminal);
    CodexRolloutEvent::new(
        "command.output",
        json!({
            "commandId": command_id,
            "toolCallId": command_id,
            "outputRef": format!("output://codex-import/{command_id}"),
            "refIds": [format!("output://codex-import/{command_id}")],
            "outputPreview": output,
            "summary": output,
            "sourceClient": "codex",
            "sourceEventType": "synthetic_command_output",
            "importedSynthetic": true,
        }),
    )
}

fn synthesized_command_exit(command_id: &str, terminal: &CodexToolCall) -> CodexRolloutEvent {
    let output = tool_output_text(terminal).unwrap_or_default();
    let exit_code = parse_exit_code(&output).unwrap_or_else(|| {
        if terminal.phase == CodexToolPhase::Failed {
            1
        } else {
            0
        }
    });
    CodexRolloutEvent::new(
        "command.exited",
        json!({
            "commandId": command_id,
            "toolCallId": command_id,
            "exitCode": exit_code,
            "result": if exit_code == 0 { "passed" } else { "failed" },
            "sourceClient": "codex",
            "sourceEventType": "synthetic_command_exited",
            "importedSynthetic": true,
        }),
    )
}

pub(super) fn tool_output_text(tool: &CodexToolCall) -> Option<String> {
    tool.output
        .as_ref()
        .and_then(visible_tool_output_text)
        .or_else(|| tool.source.output_preview.clone())
}

pub(in crate::runtime::conversation_import) fn build_canonical_history_events(
    events: Vec<CodexRolloutEvent>,
    session_id: &str,
    thread_id: &str,
    turn_id: &str,
    started_at: Option<&str>,
    completed_at: Option<&str>,
) -> Vec<RuntimeEvent> {
    let mut builder = CodexHistoryBuilder::new();
    let mut normalized = Vec::new();
    normalized.push(CodexRolloutEvent::new(
        "turn.started",
        json!({
            "startedAt": started_at,
            "imported": true,
            "sourceClient": "codex",
            "sourceEventType": "synthetic_turn_started",
            "importedSynthetic": true,
        }),
    ));
    for event in events {
        normalized.extend(builder.push(event));
    }
    let has_terminal_event = builder.has_terminal_event();
    normalized.extend(builder.finish());
    if !has_terminal_event {
        normalized.push(CodexRolloutEvent::new(
            "turn.completed",
            json!({
                "completedAt": completed_at,
                "imported": true,
                "sourceClient": "codex",
            }),
        ));
    }
    if let Some(completed_at) = completed_at {
        for event in &mut normalized {
            attach_imported_terminal_timestamp(event, completed_at);
        }
    }

    project_rollout_events_to_canonical(&normalized, session_id, thread_id, turn_id)
        .into_iter()
        .map(|event| {
            let (event_type, payload) = event
                .into_runtime()
                .expect("Codex history builder must emit canonical runtime events");
            RuntimeEvent::new(event_type, enrich_history_payload(payload))
        })
        .collect()
}

fn attach_imported_terminal_timestamp(event: &mut CodexRolloutEvent, completed_at: &str) {
    let CodexRolloutEvent::Runtime {
        event_type,
        payload,
    } = event
    else {
        return;
    };
    if !matches!(
        *event_type,
        "turn.completed" | "turn.failed" | "turn.canceled"
    ) {
        return;
    }
    let Some(payload) = payload.as_object_mut() else {
        return;
    };
    payload
        .entry("completedAt".to_string())
        .or_insert_with(|| Value::String(completed_at.to_string()));
}

fn historical_action_resolved(action_id: &str) -> CodexRolloutEvent {
    CodexRolloutEvent::new(
        "action.resolved",
        json!({
            "actionId": action_id,
            "requestId": action_id,
            "actionType": "tool_confirmation",
            "sourceClient": "codex",
            "sourceEventType": "synthetic_action_resolved",
            "importedSynthetic": true,
            "importedReadOnly": true,
        }),
    )
}

fn string_payload(payload: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| payload.get(*key).and_then(Value::as_str))
        .map(str::to_string)
        .filter(|value| !value.trim().is_empty())
}

fn parse_exit_code(output: &str) -> Option<i64> {
    output.lines().find_map(|line| {
        line.strip_prefix("Exit code:")
            .and_then(|value| value.trim().parse::<i64>().ok())
    })
}
