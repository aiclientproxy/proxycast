use crate::RuntimeEvent;
use lime_agent::AgentToolResult;
use patch_apply::{
    parse_patch, Hunk, ADD_FILE_MARKER, BEGIN_PATCH_MARKER, DELETE_FILE_MARKER, END_PATCH_MARKER,
    UPDATE_FILE_MARKER,
};
use serde_json::{json, Map, Value};

use super::{
    command_from_arguments, compact_object, dedupe_non_empty, lookup_key, non_empty_string,
    stable_scope_id, TrackedTool,
};

pub(super) fn patch_id_for_tool_start(
    tool_name: &str,
    tool_id: &str,
    arguments: Option<&Value>,
) -> Option<String> {
    if is_apply_patch_tool(tool_name)
        || command_text_from_arguments(arguments)
            .is_some_and(|command| is_apply_patch_command(&command))
    {
        Some(stable_scope_id("patch", tool_id))
    } else {
        None
    }
}

pub(super) fn patch_terminal_events(
    tool_id: &str,
    tool: &TrackedTool,
    result: &AgentToolResult,
) -> Vec<RuntimeEvent> {
    let Some(patch_id) = &tool.patch_id else {
        return Vec::new();
    };
    let changes = patch_changes_from_result(result).or_else(|| {
        let changes = patch_changes_from_arguments(tool.arguments.as_ref());
        (!changes.is_empty()).then_some(changes)
    });
    if result.success {
        vec![RuntimeEvent::new(
            "patch.applied",
            compact_object(json!({
                "patchId": patch_id,
                "toolCallId": tool_id,
                "toolName": tool.name,
                "changes": changes,
                "source": "runtime_tool",
            })),
        )]
    } else {
        let error = result
            .error
            .as_deref()
            .and_then(non_empty_string)
            .or_else(|| non_empty_string(&result.output));
        let declined = patch_declined(result);
        vec![RuntimeEvent::new(
            if declined {
                "patch.declined"
            } else {
                "patch.failed"
            },
            compact_object(json!({
                "patchId": patch_id,
                "toolCallId": tool_id,
                "toolName": tool.name,
                "status": if declined { "declined" } else { "failed" },
                "failureCategory": if declined { "approval_declined" } else { patch_failure_category(tool, result) },
                "error": error,
                "changes": changes,
                "source": "runtime_tool",
            })),
        )]
    }
}

pub(super) fn patch_changes_from_arguments(arguments: Option<&Value>) -> Vec<Value> {
    let Some(patch_text) = patch_text_from_arguments(arguments) else {
        return Vec::new();
    };
    let Ok(parsed) = parse_patch(&patch_text) else {
        return Vec::new();
    };
    parsed
        .hunks
        .into_iter()
        .map(|hunk| match hunk {
            Hunk::AddFile { path, contents } => json!({
                "kind": "add",
                "path": path.to_string_lossy(),
                "diff": contents,
            }),
            Hunk::DeleteFile { path } => json!({
                "kind": "delete",
                "path": path.to_string_lossy(),
                "diff": "",
            }),
            Hunk::UpdateFile {
                path,
                move_path,
                chunks,
            } => json!({
                "kind": "update",
                "path": path.to_string_lossy(),
                "movePath": move_path.map(|path| path.to_string_lossy().into_owned()),
                "diff": format_update_diff(&chunks),
            }),
        })
        .collect()
}

pub(super) fn patch_paths_from_arguments(arguments: Option<&Value>) -> Option<Vec<String>> {
    let patch_text = patch_text_from_arguments(arguments)?;
    if let Ok(parsed) = parse_patch(&patch_text) {
        return dedupe_non_empty(
            parsed
                .hunks
                .iter()
                .map(|hunk| hunk.path().display().to_string())
                .collect(),
        );
    }

    let paths = patch_text
        .lines()
        .filter_map(patch_path_from_line)
        .collect::<Vec<_>>();
    (!paths.is_empty()).then_some(paths)
}

fn patch_failure_category(tool: &TrackedTool, result: &AgentToolResult) -> &'static str {
    if let Some(patch_text) = patch_text_from_arguments(tool.arguments.as_ref()) {
        if parse_patch(&patch_text).is_err() {
            return "invalid_patch";
        }
    }

    let message = result
        .error
        .as_deref()
        .unwrap_or(&result.output)
        .to_ascii_lowercase();
    if message.contains("conflict") {
        "conflict"
    } else if message.contains("not found") || message.contains("no such file") {
        "missing_target"
    } else if message.contains("permission") || message.contains("denied") {
        "permission_denied"
    } else if message.contains("parse") || message.contains("invalid") {
        "invalid_patch"
    } else {
        "execution_failed"
    }
}

pub(super) fn patch_declined(result: &AgentToolResult) -> bool {
    let metadata = result.metadata.as_ref();
    metadata
        .and_then(|metadata| metadata.get("reasonCode"))
        .and_then(Value::as_str)
        == Some("tool_approval_declined")
        || metadata
            .and_then(|metadata| metadata.get("tool_outcome"))
            .and_then(Value::as_str)
            == Some("aborted")
}

fn patch_changes_from_result(result: &AgentToolResult) -> Option<Vec<Value>> {
    let changes = result
        .metadata
        .as_ref()?
        .get("file_changes")?
        .get("changes")?
        .as_array()?;
    let changes = changes
        .iter()
        .filter_map(normalize_result_change)
        .collect::<Vec<_>>();
    (!changes.is_empty()).then_some(changes)
}

fn normalize_result_change(change: &Value) -> Option<Value> {
    let change = change.as_object()?;
    let kind = change.get("kind")?.as_str()?.to_ascii_lowercase();
    let source_path = value_string_from_object(change, &["sourcePath", "source_path"]);
    let destination_path = value_string_from_object(change, &["path"])?;
    let move_path = value_string_from_object(change, &["movePath", "move_path"]);
    let (path, kind, move_path) = match kind.as_str() {
        "add" => (destination_path, "add", None),
        "delete" => (destination_path, "delete", None),
        "update" => (destination_path, "update", None),
        "move_update" | "move" => (
            source_path.unwrap_or(destination_path),
            "update",
            move_path.or_else(|| value_string_from_object(change, &["path"])),
        ),
        _ => return None,
    };
    Some(json!({
        "kind": kind,
        "path": path,
        "movePath": move_path,
        "diff": change.get("diff").map(diff_text).unwrap_or_default(),
    }))
}

fn format_update_diff(chunks: &[patch_apply::UpdateFileChunk]) -> String {
    let mut lines = Vec::new();
    for chunk in chunks {
        if let Some(context) = &chunk.change_context {
            lines.push(format!("@@ {context}"));
        }
        lines.extend(chunk.old_lines.iter().map(|line| format!("-{line}")));
        lines.extend(chunk.new_lines.iter().map(|line| format!("+{line}")));
        if chunk.is_end_of_file {
            lines.push("*** End of File".to_string());
        }
    }
    lines.join("\n")
}

fn diff_text(value: &Value) -> String {
    if let Some(text) = value.as_str() {
        return text.to_string();
    }
    value
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|line| {
            let kind = line.get("kind")?.as_str()?;
            let value = line
                .get("value")
                .and_then(Value::as_str)
                .unwrap_or_default();
            match kind {
                "add" => Some(format!("+{value}")),
                "remove" => Some(format!("-{value}")),
                "context" => Some(format!(" {value}")),
                "truncated" => Some("...".to_string()),
                _ => None,
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn value_string_from_object(value: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        value
            .get(*key)
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(str::to_string)
    })
}

fn is_apply_patch_tool(tool_name: &str) -> bool {
    matches!(
        lookup_key(tool_name).as_str(),
        "applypatch" | "applypatchtool"
    )
}

fn is_apply_patch_command(command: &str) -> bool {
    command
        .split_whitespace()
        .any(|segment| segment.trim_matches(['\'', '"']) == "apply_patch")
}

fn patch_text_from_arguments(arguments: Option<&Value>) -> Option<String> {
    let arguments = arguments?;
    value_text_from_object(arguments, &["patch", "diff", "input", "stdin"]).or_else(|| {
        command_text_from_arguments(Some(arguments)).and_then(|command| {
            if is_apply_patch_command(&command) {
                extract_patch_block(&command)
            } else {
                None
            }
        })
    })
}

fn command_text_from_arguments(arguments: Option<&Value>) -> Option<String> {
    command_from_arguments(arguments).or_else(|| {
        let value = arguments?;
        value_text_from_object(value, &["command", "cmd", "script"])
    })
}

fn value_text_from_object(value: &Value, keys: &[&str]) -> Option<String> {
    let object = value.as_object()?;
    keys.iter()
        .find_map(|key| object.get(*key))
        .and_then(value_text)
}

fn value_text(value: &Value) -> Option<String> {
    value.as_str().and_then(non_empty_string).or_else(|| {
        let values = value.as_array()?;
        let joined = values
            .iter()
            .filter_map(|value| value.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        non_empty_string(&joined)
    })
}

fn extract_patch_block(value: &str) -> Option<String> {
    let start = value.find(BEGIN_PATCH_MARKER)?;
    let patch_text = &value[start..];
    let end = patch_text.find(END_PATCH_MARKER)? + END_PATCH_MARKER.len();
    Some(patch_text[..end].to_string())
}

fn patch_path_from_line(line: &str) -> Option<String> {
    let trimmed = line.trim();
    for prefix in [
        ADD_FILE_MARKER,
        UPDATE_FILE_MARKER,
        DELETE_FILE_MARKER,
        "*** Move to: ",
    ] {
        if let Some(path) = trimmed.strip_prefix(prefix) {
            return non_empty_string(path);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_paths_with_parser_and_move_destination() {
        let arguments = json!({
            "patch": "*** Begin Patch\n*** Add File: src/new.rs\n+new\n*** Update File: src/old.rs\n*** Move to: src/moved.rs\n@@\n-old\n+new\n*** End Patch\n",
        });

        assert_eq!(
            patch_paths_from_arguments(Some(&arguments)),
            Some(vec!["src/new.rs".to_string(), "src/moved.rs".to_string()])
        );
    }

    #[test]
    fn extracts_patch_block_from_shell_command() {
        let arguments = json!({
            "command": "apply_patch <<'PATCH'\n*** Begin Patch\n*** Add File: notes/live.md\n+hello\n*** End Patch\nPATCH",
        });

        assert_eq!(
            patch_paths_from_arguments(Some(&arguments)),
            Some(vec!["notes/live.md".to_string()])
        );
    }

    #[test]
    fn extracts_canonical_add_delete_update_and_move_changes() {
        let arguments = json!({
            "patch": "*** Begin Patch\n*** Add File: src/new.rs\n+new\n*** Delete File: src/dead.rs\n*** Update File: src/current.rs\n@@\n-old\n+new\n*** Update File: src/source.rs\n*** Move to: src/target.rs\n@@\n-before\n+after\n*** End Patch\n",
        });

        let changes = patch_changes_from_arguments(Some(&arguments));
        assert_eq!(changes.len(), 4);
        assert_eq!(changes[0]["kind"], "add");
        assert_eq!(changes[1]["kind"], "delete");
        assert_eq!(changes[2]["kind"], "update");
        assert_eq!(changes[3]["path"], "src/source.rs");
        assert_eq!(changes[3]["movePath"], "src/target.rs");
    }

    #[test]
    fn invalid_arguments_patch_overrides_generic_failure_category() {
        let tool = TrackedTool {
            name: "apply_patch".to_string(),
            arguments: Some(json!({
                "patch": "*** Begin Patch\n*** Update File: broken.rs\n*** End Patch",
            })),
            command_facts: None,
            test_run_id: None,
            patch_id: Some("patch:1".to_string()),
            emitted_output: false,
        };
        let result = AgentToolResult {
            success: false,
            output: "target file not found".to_string(),
            error: None,
            structured_content: None,
            images: None,
            metadata: None,
        };

        assert_eq!(patch_failure_category(&tool, &result), "invalid_patch");
    }
}
