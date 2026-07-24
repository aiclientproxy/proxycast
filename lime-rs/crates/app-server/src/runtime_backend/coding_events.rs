use crate::RuntimeEvent;
use agent_protocol::{ItemStatus, ThreadItem, ThreadItemPayload, ToolArgument, ToolOutput};
use lime_agent::{AgentEvent as RuntimeAgentEvent, AgentToolResult};
use serde_json::{json, Value};
use std::collections::HashMap;

mod command;
mod patch;

use command::{command_facts_from_arguments, command_facts_from_text, CommandFacts};
use patch::{
    patch_changes_from_arguments, patch_id_for_tool_start, patch_paths_from_arguments,
    patch_terminal_events,
};

const COMMAND_OUTPUT_PREVIEW_CHARS: usize = 1_200;

#[derive(Debug, Clone)]
struct TrackedTool {
    name: String,
    arguments: Option<Value>,
    command_facts: Option<CommandFacts>,
    test_run_id: Option<String>,
    patch_id: Option<String>,
    emitted_output: bool,
}

#[derive(Debug, Default)]
pub(super) struct CodingEventMirror {
    tools: HashMap<String, TrackedTool>,
}

#[derive(Debug, Default)]
pub(super) struct CodingMirrorEvents {
    pub(super) before_raw: Vec<RuntimeEvent>,
    pub(super) after_raw: Vec<RuntimeEvent>,
}

impl CodingEventMirror {
    pub(super) fn process_event(&mut self, event: &RuntimeAgentEvent) -> CodingMirrorEvents {
        match event {
            RuntimeAgentEvent::ItemStarted { item } => CodingMirrorEvents {
                after_raw: canonical_tool_item(item)
                    .map(|tool| {
                        self.handle_tool_start(
                            tool.name,
                            tool.call_id,
                            canonical_arguments_value(tool.arguments).as_ref(),
                        )
                    })
                    .unwrap_or_default(),
                ..CodingMirrorEvents::default()
            },
            RuntimeAgentEvent::ToolOutputDelta {
                tool_id,
                delta,
                output_kind,
                metadata,
            } => CodingMirrorEvents {
                after_raw: self.handle_tool_output_delta(
                    tool_id,
                    delta,
                    output_kind.as_deref(),
                    metadata.as_ref(),
                ),
                ..CodingMirrorEvents::default()
            },
            RuntimeAgentEvent::ItemCompleted { item } => canonical_tool_item(item)
                .and_then(|tool| {
                    tool.output.map(|output| {
                        let result = canonical_agent_tool_result(item, output);
                        self.handle_tool_end(tool.call_id, &result)
                    })
                })
                .unwrap_or_default(),
            _ => CodingMirrorEvents::default(),
        }
    }

    fn handle_tool_start(
        &mut self,
        tool_name: &str,
        tool_id: &str,
        arguments: Option<&Value>,
    ) -> Vec<RuntimeEvent> {
        let normalized_name = normalize_tool_name(tool_name);
        let arguments_value = arguments.cloned();
        let command_facts = command_facts_from_arguments(arguments_value.as_ref());
        let mut events = Vec::new();
        let patch_id = patch_id_for_tool_start(normalized_name, tool_id, arguments_value.as_ref());
        if let Some(patch_id) = &patch_id {
            events.push(RuntimeEvent::new(
                "patch.started",
                compact_object(json!({
                    "patchId": patch_id,
                    "toolCallId": tool_id,
                    "toolName": normalized_name,
                    "source": "runtime_tool",
                    "changes": patch_changes_from_arguments(arguments_value.as_ref()),
                    "paths": patch_paths_from_arguments(arguments_value.as_ref()),
                })),
            ));
        }
        let test_run_id = if is_shell_tool(normalized_name) {
            let command = command_facts
                .as_ref()
                .map(|facts| facts.command.clone())
                .unwrap_or_default();
            events.push(RuntimeEvent::new(
                "command.started",
                compact_object(json!({
                    "commandId": tool_id,
                    "toolCallId": tool_id,
                    "toolName": normalized_name,
                    "command": command,
                    "canonicalCommand": command_facts.as_ref().map(|facts| facts.canonical_command.clone()),
                    "commandSummary": command_facts.as_ref().map(|facts| facts.summary.clone()),
                    "commandArgv": command_facts.as_ref().map(|facts| facts.argv.clone()),
                    "commandArgvSource": command_facts.as_ref().map(|facts| facts.source),
                    "cwd": cwd_from_value(arguments_value.as_ref()),
                    "source": "runtime_tool",
                })),
            ));

            if is_likely_test_command(&command) {
                let test_run_id = stable_scope_id("test", tool_id);
                events.push(RuntimeEvent::new(
                    "test.started",
                    compact_object(json!({
                        "testRunId": test_run_id,
                        "commandId": tool_id,
                        "command": command,
                        "canonicalCommand": command_facts.as_ref().map(|facts| facts.canonical_command.clone()),
                        "commandSummary": command_facts.as_ref().map(|facts| facts.summary.clone()),
                        "source": "runtime_tool",
                    })),
                ));
                Some(test_run_id)
            } else {
                None
            }
        } else {
            None
        };

        self.tools.insert(
            tool_id.to_string(),
            TrackedTool {
                name: normalized_name.to_string(),
                arguments: arguments_value,
                command_facts,
                test_run_id,
                patch_id,
                emitted_output: false,
            },
        );

        events
    }

    fn handle_tool_output_delta(
        &mut self,
        tool_id: &str,
        delta: &str,
        output_kind: Option<&str>,
        metadata: Option<&HashMap<String, Value>>,
    ) -> Vec<RuntimeEvent> {
        let Some(tool) = self.tools.get_mut(tool_id) else {
            return Vec::new();
        };
        let has_delta = !delta.is_empty();
        let has_process_metadata = metadata_has_process_lifecycle(metadata);
        if !is_shell_tool(&tool.name) || (!has_delta && !has_process_metadata) {
            return Vec::new();
        }

        if has_delta {
            tool.emitted_output = true;
        }
        let output_ref = output_ref_from_metadata(metadata, "command")
            .unwrap_or_else(|| command_output_ref(tool_id));
        let ref_ids = output_ref_ids(metadata, &output_ref);
        let preview = has_delta.then(|| truncate_chars(delta, COMMAND_OUTPUT_PREVIEW_CHARS));
        vec![RuntimeEvent::new(
            "command.output",
            compact_object(json!({
                "commandId": tool_id,
                "toolCallId": tool_id,
                "outputRef": output_ref,
                "refIds": ref_ids,
                "kind": output_kind,
                "delta": has_delta.then(|| delta.to_string()),
                "preview": preview,
                "source": "runtime_tool_stream",
                "metadata": metadata.cloned(),
            })),
        )]
    }

    fn handle_tool_end(&mut self, tool_id: &str, result: &AgentToolResult) -> CodingMirrorEvents {
        let Some(tool) = self.tools.remove(tool_id) else {
            return CodingMirrorEvents::default();
        };
        let after_raw = match tool.name.as_str() {
            "Bash" | "PowerShell" => self.shell_tool_end_events(tool_id, &tool, result),
            "Read" => file_read_tool_end_events(tool_id, &tool, result),
            "Write" | "Edit" | "apply_patch" => file_tool_end_events(tool_id, &tool, result),
            _ => Vec::new(),
        };
        CodingMirrorEvents {
            before_raw: policy_block_events(tool_id, &tool, result),
            after_raw,
        }
    }

    fn shell_tool_end_events(
        &self,
        tool_id: &str,
        tool: &TrackedTool,
        result: &AgentToolResult,
    ) -> Vec<RuntimeEvent> {
        let metadata = result.metadata.as_ref();
        let command_text = metadata
            .and_then(|metadata| metadata_string(metadata, &["command"]))
            .or_else(|| command_from_arguments(tool.arguments.as_ref()));
        let command_facts = command_text
            .as_deref()
            .and_then(command_facts_from_text)
            .or_else(|| tool.command_facts.clone());
        let command =
            command_text.or_else(|| command_facts.as_ref().map(|facts| facts.command.clone()));
        let exit_code = metadata.and_then(|metadata| metadata_i64(metadata, &["exit_code"]));
        let status = command_status(exit_code, result.success);
        let process_metadata = shell_process_lifecycle_metadata(tool_id, metadata, status);
        let mut events = Vec::new();

        if !tool.emitted_output && !result.output.is_empty() {
            let output_ref = output_ref_from_metadata(metadata, "command")
                .unwrap_or_else(|| command_output_ref(tool_id));
            let ref_ids = output_ref_ids(metadata, &output_ref);
            events.push(RuntimeEvent::new(
                "command.output",
                compact_object(merge_object_fields(
                    json!({
                        "commandId": tool_id,
                        "toolCallId": tool_id,
                        "outputRef": output_ref,
                        "refIds": ref_ids,
                        "delta": result.output,
                        "preview": truncate_chars(&result.output, COMMAND_OUTPUT_PREVIEW_CHARS),
                        "source": "runtime_tool_result",
                        "metadata": process_metadata.clone(),
                    }),
                    &process_metadata,
                )),
            ));
        }

        events.push(RuntimeEvent::new(
            "command.exited",
            compact_object(merge_object_fields(json!({
                "commandId": tool_id,
                "toolCallId": tool_id,
                "command": command,
                "canonicalCommand": command_facts.as_ref().map(|facts| facts.canonical_command.clone()),
                "commandSummary": command_facts.as_ref().map(|facts| facts.summary.clone()),
                "commandArgv": command_facts.as_ref().map(|facts| facts.argv.clone()),
                "commandArgvSource": command_facts.as_ref().map(|facts| facts.source),
                "exitCode": exit_code,
                "status": status,
                "success": result.success,
                "cwd": metadata.and_then(|metadata| metadata_string(metadata, &["cwd"])),
                "shell": metadata.and_then(|metadata| metadata_string(metadata, &["shell"])),
                "source": "runtime_tool",
            }), &process_metadata)),
        ));
        events.extend(patch_terminal_events(tool_id, tool, result));

        if let Some(test_run_id) = &tool.test_run_id {
            events.push(RuntimeEvent::new(
                "test.completed",
                compact_object(json!({
                    "testRunId": test_run_id,
                    "commandId": tool_id,
                    "command": command,
                    "canonicalCommand": command_facts.as_ref().map(|facts| facts.canonical_command.clone()),
                    "commandSummary": command_facts.as_ref().map(|facts| facts.summary.clone()),
                    "result": if status == "passed" { "passed" } else { "failed" },
                    "status": status,
                    "exitCode": exit_code,
                    "source": "runtime_tool",
                })),
            ));
        }

        events
    }
}

struct CanonicalToolItem<'a> {
    call_id: &'a str,
    name: &'a str,
    arguments: &'a [ToolArgument],
    output: Option<&'a ToolOutput>,
}

fn canonical_tool_item(item: &ThreadItem) -> Option<CanonicalToolItem<'_>> {
    let ThreadItemPayload::Tool {
        call_id,
        name,
        arguments,
        output,
    } = &item.payload
    else {
        return None;
    };
    Some(CanonicalToolItem {
        call_id,
        name,
        arguments,
        output: output.as_ref(),
    })
}

fn canonical_arguments_value(arguments: &[ToolArgument]) -> Option<Value> {
    if arguments.is_empty() {
        return None;
    }
    if let [argument] = arguments {
        if argument.name == "value" {
            return Some(
                serde_json::from_str(&argument.value)
                    .unwrap_or_else(|_| Value::String(argument.value.clone())),
            );
        }
    }
    Some(Value::Object(
        arguments
            .iter()
            .map(|argument| {
                let value = serde_json::from_str(&argument.value)
                    .unwrap_or_else(|_| Value::String(argument.value.clone()));
                (argument.name.clone(), value)
            })
            .collect(),
    ))
}

fn canonical_agent_tool_result(item: &ThreadItem, output: &ToolOutput) -> AgentToolResult {
    let mut metadata = item
        .metadata
        .as_object()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .collect::<HashMap<_, _>>();
    if let Some(duration_ms) = output.duration_ms {
        metadata
            .entry("duration_ms".to_string())
            .or_insert(Value::from(duration_ms));
    }
    if output.truncated {
        metadata
            .entry("truncated".to_string())
            .or_insert(Value::Bool(true));
    }
    if let Some(output_ref) = output.output_ref.as_ref() {
        metadata
            .entry("output_ref".to_string())
            .or_insert_with(|| Value::String(output_ref.clone()));
    }
    AgentToolResult {
        success: item.status == ItemStatus::Completed,
        output: output.text.clone().unwrap_or_default(),
        error: output.error.clone(),
        structured_content: output.structured_content.clone(),
        images: None,
        metadata: (!metadata.is_empty()).then_some(metadata),
    }
}

fn file_tool_end_events(
    tool_id: &str,
    tool: &TrackedTool,
    result: &AgentToolResult,
) -> Vec<RuntimeEvent> {
    let mut events = patch_terminal_events(tool_id, tool, result);
    if tool.patch_id.is_some() {
        return events;
    }
    if !result.success {
        return events;
    }

    let Some(path) = file_path_from_result(result, tool.arguments.as_ref()) else {
        return events;
    };
    let artifact_id = result
        .metadata
        .as_ref()
        .and_then(|metadata| metadata_string(metadata, &["artifact_id", "artifactId"]))
        .unwrap_or_else(|| stable_scope_id("artifact:file", &path));
    let metadata = result.metadata.as_ref();
    let artifact_refs = artifact_refs_from_metadata(metadata, &artifact_id);

    events.push(RuntimeEvent::new(
        "file.changed",
        compact_object(json!({
            "path": path,
            "artifactId": artifact_id,
            "artifactRefs": artifact_refs,
            "toolCallId": tool_id,
            "toolName": tool.name,
            "checkpointRef": metadata.and_then(|metadata| metadata_string(metadata, &["checkpointRef", "checkpoint_ref", "checkpointId", "checkpoint_id"])),
            "contentRef": metadata.and_then(|metadata| metadata_string(metadata, &["contentRef", "content_ref"])),
            "diffRef": metadata.and_then(|metadata| metadata_string(metadata, &["diffRef", "diff_ref"])),
            "preview": metadata.and_then(|metadata| metadata_string(metadata, &["preview", "summary", "previewText", "preview_text"])),
            "change": result
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.get("file_change").cloned()),
            "source": "runtime_tool",
        })),
    ));
    events
}

fn file_read_tool_end_events(
    tool_id: &str,
    tool: &TrackedTool,
    result: &AgentToolResult,
) -> Vec<RuntimeEvent> {
    if !result.success {
        return Vec::new();
    }

    let Some(path) = file_path_from_result(result, tool.arguments.as_ref()) else {
        return Vec::new();
    };
    let metadata = result.metadata.as_ref();
    let output_ref =
        output_ref_from_metadata(metadata, "file").unwrap_or_else(|| file_output_ref(tool_id));
    let ref_ids = output_ref_ids(metadata, &output_ref);

    vec![RuntimeEvent::new(
        "file.read",
        compact_object(json!({
            "path": path,
            "toolCallId": tool_id,
            "toolName": tool.name,
            "outputRef": output_ref,
            "contentRef": metadata.and_then(|metadata| metadata_string(metadata, &["contentRef", "content_ref"])),
            "refIds": ref_ids,
            "startLine": value_number_from_object(tool.arguments.as_ref(), &["start_line", "startLine"]),
            "endLine": value_number_from_object(tool.arguments.as_ref(), &["end_line", "endLine"]),
            "source": "runtime_tool",
            "fileType": result
                .metadata
                .as_ref()
                .and_then(|metadata| metadata_string(metadata, &["file_type", "fileType"])),
        })),
    )]
}

fn policy_block_events(
    tool_id: &str,
    tool: &TrackedTool,
    result: &AgentToolResult,
) -> Vec<RuntimeEvent> {
    if result.success {
        return Vec::new();
    }
    let diagnostics = policy_diagnostics(tool, result);

    match policy_block_kind(result) {
        Some(PolicyBlockKind::PermissionDenied) => vec![RuntimeEvent::new(
            "permission.denied",
            compact_object(json!({
                "toolCallId": tool_id,
                "toolName": tool.name,
                "reasonCode": policy_reason_code(result, "permission_denied"),
                "reason": policy_reason(result),
                "policyName": diagnostics.get("policyName").cloned(),
                "policyProfile": diagnostics.get("policyProfile").cloned(),
                "policyDecisionId": diagnostics.get("policyDecisionId").cloned(),
                "platform": diagnostics.get("platform").cloned(),
                "command": diagnostics.get("command").cloned(),
                "cwd": diagnostics.get("cwd").cloned(),
                "diagnostics": diagnostics,
                "source": "runtime_tool",
            })),
        )],
        Some(PolicyBlockKind::SandboxBlocked) => vec![RuntimeEvent::new(
            "sandbox.blocked",
            compact_object(json!({
                "toolCallId": tool_id,
                "toolName": tool.name,
                "reasonCode": policy_reason_code(result, "sandbox_blocked"),
                "reason": policy_reason(result),
                "sandboxPolicy": diagnostics.get("sandboxPolicy").cloned(),
                "policyProfile": diagnostics.get("policyProfile").cloned(),
                "policyDecisionId": diagnostics.get("policyDecisionId").cloned(),
                "platform": diagnostics.get("platform").cloned(),
                "command": diagnostics.get("command").cloned(),
                "cwd": diagnostics.get("cwd").cloned(),
                "diagnostics": diagnostics,
                "source": "runtime_tool",
            })),
        )],
        None => Vec::new(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PolicyBlockKind {
    PermissionDenied,
    SandboxBlocked,
}

fn policy_block_kind(result: &AgentToolResult) -> Option<PolicyBlockKind> {
    let metadata = result.metadata.as_ref();
    if metadata
        .and_then(|metadata| metadata_string(metadata, &["eventClass", "event_class"]))
        .as_deref()
        == Some("sandbox.blocked")
    {
        return Some(PolicyBlockKind::SandboxBlocked);
    }
    if metadata
        .and_then(|metadata| metadata_string(metadata, &["eventClass", "event_class"]))
        .as_deref()
        == Some("permission.denied")
    {
        return Some(PolicyBlockKind::PermissionDenied);
    }

    let category = metadata
        .and_then(|metadata| {
            metadata_string(
                metadata,
                &[
                    "failureCategory",
                    "failure_category",
                    "reasonCode",
                    "reason_code",
                    "code",
                ],
            )
        })
        .unwrap_or_default()
        .to_ascii_lowercase();
    if category.contains("sandbox") {
        return Some(PolicyBlockKind::SandboxBlocked);
    }
    if category.contains("permission") || category.contains("denied") || category.contains("policy")
    {
        return Some(PolicyBlockKind::PermissionDenied);
    }

    let text = format!(
        "{}\n{}",
        result.error.as_deref().unwrap_or_default(),
        result.output
    )
    .to_ascii_lowercase();
    if text.contains("sandbox") && (text.contains("block") || text.contains("denied")) {
        Some(PolicyBlockKind::SandboxBlocked)
    } else if text.contains("permission denied")
        || text.contains("access denied")
        || text.contains("policy denied")
        || text.contains("not allowed")
    {
        Some(PolicyBlockKind::PermissionDenied)
    } else {
        None
    }
}

fn policy_reason_code(result: &AgentToolResult, fallback: &str) -> String {
    result
        .metadata
        .as_ref()
        .and_then(|metadata| metadata_string(metadata, &["reasonCode", "reason_code", "code"]))
        .unwrap_or_else(|| fallback.to_string())
}

fn policy_reason(result: &AgentToolResult) -> Option<String> {
    result
        .metadata
        .as_ref()
        .and_then(|metadata| metadata_string(metadata, &["reason", "message"]))
        .or_else(|| result.error.as_deref().and_then(non_empty_string))
        .or_else(|| non_empty_string(&result.output))
}

fn policy_diagnostics(tool: &TrackedTool, result: &AgentToolResult) -> Value {
    let metadata = result.metadata.as_ref();
    compact_object(json!({
        "policyName": metadata.and_then(|metadata| metadata_string(metadata, &["policyName", "policy_name", "policy"])),
        "policyProfile": metadata.and_then(|metadata| metadata_string(metadata, &["policyProfile", "policy_profile", "profile"])),
        "policyDecisionId": metadata.and_then(|metadata| metadata_string(metadata, &["policyDecisionId", "policy_decision_id", "decisionId", "decision_id"])),
        "sandboxPolicy": metadata.and_then(|metadata| metadata_string(metadata, &["sandboxPolicy", "sandbox_policy", "sandbox"])),
        "sandboxReason": metadata.and_then(|metadata| metadata_string(metadata, &["sandboxReason", "sandbox_reason"])),
        "platform": metadata.and_then(|metadata| metadata_string(metadata, &["platform", "os", "target_os"])).or_else(|| Some(std::env::consts::OS.to_string())),
        "arch": metadata.and_then(|metadata| metadata_string(metadata, &["arch", "target_arch"])).or_else(|| Some(std::env::consts::ARCH.to_string())),
        "toolSurface": metadata.and_then(|metadata| metadata_string(metadata, &["toolSurface", "tool_surface"])).unwrap_or_else(|| "runtime_tool".to_string()),
        "toolName": tool.name,
        "command": command_from_arguments(tool.arguments.as_ref()).or_else(|| metadata.and_then(|metadata| metadata_string(metadata, &["command", "cmd", "script"]))),
        "canonicalCommand": tool.command_facts.as_ref().map(|facts| facts.canonical_command.clone()),
        "commandSummary": tool.command_facts.as_ref().map(|facts| facts.summary.clone()),
        "commandArgv": tool.command_facts.as_ref().map(|facts| facts.argv.clone()),
        "commandArgvSource": tool.command_facts.as_ref().map(|facts| facts.source),
        "cwd": cwd_from_value(tool.arguments.as_ref()).or_else(|| metadata.and_then(|metadata| metadata_string(metadata, &["cwd", "workingDir", "working_dir"]))),
    }))
}

fn normalize_tool_name(tool_name: &str) -> &str {
    match lookup_key(tool_name).as_str() {
        "bashtool" | "shell" | "developershell" | "mcpsystemshell" | "shellcommand"
        | "execcommand" | "localshellcall" => "Bash",
        "powershelltool" => "PowerShell",
        "filewritetool" | "writefiletool" | "createfiletool" | "writefile" | "createfile"
        | "mcpsystemwritefile" => "Write",
        "fileedittool" | "editfile" | "developertexteditor" | "mcpsystemeditfile" => "Edit",
        "filereadtool" | "readfiletool" | "readfile" | "developerread" | "mcpsystemreadfile" => {
            "Read"
        }
        "applypatch" | "applypatchtool" => "apply_patch",
        _ => tool_name.trim(),
    }
}

fn lookup_key(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .map(|character| character.to_ascii_lowercase())
        .collect()
}

fn is_shell_tool(tool_name: &str) -> bool {
    matches!(tool_name, "Bash" | "PowerShell")
}

fn command_from_arguments(arguments: Option<&Value>) -> Option<String> {
    command_facts_from_arguments(arguments)
        .map(|facts| facts.command)
        .or_else(|| value_string_from_object(arguments?, &["command", "cmd", "script"]))
}

fn cwd_from_value(value: Option<&Value>) -> Option<String> {
    value_string_from_object(value?, &["cwd", "workingDir", "working_dir"])
}

fn file_path_from_result(result: &AgentToolResult, arguments: Option<&Value>) -> Option<String> {
    result
        .metadata
        .as_ref()
        .and_then(|metadata| {
            metadata_string(
                metadata,
                &[
                    "path",
                    "file_path",
                    "filePath",
                    "artifact_path",
                    "artifactPath",
                ],
            )
        })
        .or_else(|| {
            result
                .metadata
                .as_ref()
                .and_then(|metadata| metadata_paths(metadata, "artifact_paths"))
                .and_then(|paths| paths.into_iter().next())
        })
        .or_else(|| value_string_from_object(arguments?, &["path", "filePath", "file_path"]))
}

fn metadata_string(metadata: &HashMap<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| metadata.get(*key))
        .and_then(value_string)
}

fn metadata_i64(metadata: &HashMap<String, Value>, keys: &[&str]) -> Option<i64> {
    keys.iter()
        .find_map(|key| metadata.get(*key))
        .and_then(value_i64)
}

fn metadata_u64(metadata: &HashMap<String, Value>, keys: &[&str]) -> Option<u64> {
    keys.iter()
        .find_map(|key| metadata.get(*key))
        .and_then(value_u64)
}

fn metadata_bool(metadata: &HashMap<String, Value>, keys: &[&str]) -> Option<bool> {
    keys.iter()
        .find_map(|key| metadata.get(*key))
        .and_then(value_bool)
}

fn metadata_string_array(metadata: &HashMap<String, Value>, keys: &[&str]) -> Vec<String> {
    keys.iter()
        .filter_map(|key| metadata.get(*key))
        .flat_map(value_string_vec)
        .collect()
}

fn metadata_paths(metadata: &HashMap<String, Value>, key: &str) -> Option<Vec<String>> {
    let values = metadata.get(key)?.as_array()?;
    let paths = values.iter().filter_map(value_string).collect::<Vec<_>>();
    (!paths.is_empty()).then_some(paths)
}

fn value_string_from_object(value: &Value, keys: &[&str]) -> Option<String> {
    let object = value.as_object()?;
    keys.iter()
        .find_map(|key| object.get(*key))
        .and_then(value_string)
}

fn value_number_from_object(value: Option<&Value>, keys: &[&str]) -> Option<u64> {
    let object = value?.as_object()?;
    keys.iter()
        .find_map(|key| object.get(*key))
        .and_then(value_u64)
}

fn value_string(value: &Value) -> Option<String> {
    value.as_str().and_then(non_empty_string)
}

fn value_string_vec(value: &Value) -> Vec<String> {
    if let Some(values) = value.as_array() {
        return values.iter().filter_map(value_string).collect();
    }
    value_string(value).into_iter().collect()
}

fn non_empty_string(value: &str) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn value_i64(value: &Value) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| value.as_u64().and_then(|value| i64::try_from(value).ok()))
        .or_else(|| value.as_str()?.trim().parse::<i64>().ok())
}

fn value_u64(value: &Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_i64().and_then(|value| u64::try_from(value).ok()))
        .or_else(|| value.as_str()?.trim().parse::<u64>().ok())
}

fn value_bool(value: &Value) -> Option<bool> {
    value.as_bool().or_else(|| match value.as_str()?.trim() {
        "true" | "TRUE" | "True" | "1" => Some(true),
        "false" | "FALSE" | "False" | "0" => Some(false),
        _ => None,
    })
}

fn command_status(exit_code: Option<i64>, success: bool) -> &'static str {
    match exit_code {
        Some(0) => "passed",
        Some(_) => "failed",
        None if success => "completed",
        None => "failed",
    }
}

fn is_likely_test_command(command: &str) -> bool {
    let normalized = command.to_ascii_lowercase();
    [
        "npm test",
        "npm run test",
        "pnpm test",
        "pnpm run test",
        "yarn test",
        "cargo test",
        "cargo nextest",
        "vitest",
        "jest",
        "pytest",
        "go test",
        "deno test",
        "bun test",
    ]
    .iter()
    .any(|needle| normalized.contains(needle))
}

fn command_output_ref(tool_id: &str) -> String {
    stable_scope_id("output:command", tool_id)
}

fn file_output_ref(tool_id: &str) -> String {
    stable_scope_id("output:file", tool_id)
}

fn output_ref_from_metadata(
    metadata: Option<&HashMap<String, Value>>,
    fallback_kind: &str,
) -> Option<String> {
    let metadata = metadata?;
    metadata_string(
        metadata,
        &[
            "outputRef",
            "output_ref",
            "contentRef",
            "content_ref",
            "refId",
            "ref_id",
            "outputId",
            "output_id",
        ],
    )
    .or_else(|| {
        metadata_string_array(
            metadata,
            &[
                "refIds",
                "ref_ids",
                "outputRefs",
                "output_refs",
                "contentRefs",
                "content_refs",
            ],
        )
        .into_iter()
        .next()
    })
    .or_else(|| {
        metadata_string(metadata, &["artifact_path", "artifactPath"])
            .map(|path| stable_scope_id(&format!("output:{fallback_kind}"), &path))
    })
}

fn output_ref_ids(
    metadata: Option<&HashMap<String, Value>>,
    output_ref: &str,
) -> Option<Vec<String>> {
    let mut refs = Vec::new();
    if let Some(metadata) = metadata {
        refs.extend(metadata_string_array(
            metadata,
            &[
                "refIds",
                "ref_ids",
                "outputRefs",
                "output_refs",
                "contentRefs",
                "content_refs",
            ],
        ));
        refs.extend(
            [
                "outputRef",
                "output_ref",
                "contentRef",
                "content_ref",
                "refId",
                "ref_id",
                "outputId",
                "output_id",
            ]
            .iter()
            .filter_map(|key| metadata.get(*key))
            .filter_map(value_string),
        );
    }
    refs.push(output_ref.to_string());
    dedupe_non_empty(refs)
}

fn shell_process_lifecycle_metadata(
    tool_id: &str,
    metadata: Option<&HashMap<String, Value>>,
    status: &str,
) -> Value {
    let process_id = metadata
        .and_then(|metadata| metadata_string(metadata, &["processId", "process_id"]))
        .unwrap_or_else(|| format!("process-{tool_id}"));
    let execution_process_status = metadata
        .and_then(|metadata| {
            metadata_string(
                metadata,
                &["executionProcessStatus", "execution_process_status"],
            )
        })
        .unwrap_or_else(|| command_status_to_process_status(status));
    let execution_surface = metadata
        .and_then(|metadata| metadata_string(metadata, &["executionSurface", "execution_surface"]))
        .unwrap_or_else(|| "embedded".to_string());

    compact_object(json!({
        "processId": process_id,
        "executionProcessStatus": execution_process_status,
        "executionProcessControlStatus": metadata.and_then(|metadata| metadata_string(metadata, &["executionProcessControlStatus", "execution_process_control_status"])),
        "stdinWritable": metadata.and_then(|metadata| metadata_bool(metadata, &["stdinWritable", "stdin_writable"])),
        "executionSurface": execution_surface,
        "outputBytes": metadata.and_then(|metadata| metadata_u64(metadata, &["outputBytes", "output_bytes"])),
        "outputOmittedBytes": metadata.and_then(|metadata| metadata_u64(metadata, &["outputOmittedBytes", "output_omitted_bytes"])),
        "outputTruncated": metadata.and_then(|metadata| metadata_bool(metadata, &["outputTruncated", "output_truncated"])),
        "stdoutBytes": metadata.and_then(|metadata| metadata_u64(metadata, &["stdoutBytes", "stdout_bytes"])),
        "stderrBytes": metadata.and_then(|metadata| metadata_u64(metadata, &["stderrBytes", "stderr_bytes"])),
    }))
}

fn command_status_to_process_status(status: &str) -> String {
    match status {
        "passed" | "completed" => "exited".to_string(),
        "failed" => "failed".to_string(),
        "canceled" | "cancelled" => "terminated".to_string(),
        _ => status.to_string(),
    }
}

fn metadata_has_process_lifecycle(metadata: Option<&HashMap<String, Value>>) -> bool {
    let Some(metadata) = metadata else {
        return false;
    };
    metadata_string(metadata, &["processId", "process_id"]).is_some()
        || metadata_string(
            metadata,
            &["executionProcessStatus", "execution_process_status"],
        )
        .is_some()
        || metadata_string(
            metadata,
            &[
                "executionProcessControlStatus",
                "execution_process_control_status",
            ],
        )
        .is_some()
        || metadata_bool(metadata, &["stdinWritable", "stdin_writable"]).is_some()
        || metadata_string(metadata, &["executionSurface", "execution_surface"]).is_some()
}

fn artifact_refs_from_metadata(
    metadata: Option<&HashMap<String, Value>>,
    artifact_id: &str,
) -> Option<Vec<String>> {
    let mut refs = Vec::new();
    if let Some(metadata) = metadata {
        refs.extend(metadata_string_array(
            metadata,
            &[
                "artifactRefs",
                "artifact_refs",
                "artifactIds",
                "artifact_ids",
                "artifact_ref",
                "artifactRef",
            ],
        ));
        refs.extend(
            ["artifactId", "artifact_id"]
                .iter()
                .filter_map(|key| metadata.get(*key))
                .filter_map(value_string),
        );
    }
    refs.push(artifact_id.to_string());
    dedupe_non_empty(refs)
}

fn dedupe_non_empty(values: Vec<String>) -> Option<Vec<String>> {
    let mut deduped = Vec::new();
    for value in values {
        let Some(value) = non_empty_string(&value) else {
            continue;
        };
        if !deduped.contains(&value) {
            deduped.push(value);
        }
    }
    (!deduped.is_empty()).then_some(deduped)
}

fn stable_scope_id(prefix: &str, value: &str) -> String {
    format!("{prefix}:{:016x}", stable_hash(value))
}

fn stable_hash(value: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    let mut output = String::new();
    for (index, character) in value.chars().enumerate() {
        if index >= max_chars {
            output.push('…');
            break;
        }
        output.push(character);
    }
    output
}

fn merge_object_fields(mut value: Value, fields: &Value) -> Value {
    let (Some(object), Some(fields)) = (value.as_object_mut(), fields.as_object()) else {
        return value;
    };
    for (key, value) in fields {
        object.insert(key.clone(), value.clone());
    }
    value
}

fn compact_object(value: Value) -> Value {
    match value {
        Value::Object(object) => Value::Object(
            object
                .into_iter()
                .filter_map(|(key, value)| {
                    let compacted = compact_object(value);
                    if compacted.is_null() {
                        None
                    } else {
                        Some((key, compacted))
                    }
                })
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.into_iter().map(compact_object).collect()),
        other => other,
    }
}

#[cfg(test)]
mod tests;
