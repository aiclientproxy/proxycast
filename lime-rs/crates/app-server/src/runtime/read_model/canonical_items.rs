use super::approval;
use agent_protocol::{FileChangeStatus, ItemStatus, ThreadItem, ThreadItemPayload};
use serde_json::json;

pub(in crate::runtime) fn canonical_item_to_agent_detail(item: &ThreadItem) -> serde_json::Value {
    let (item_type, payload) = canonical_payload_to_agent_detail(&item.payload);
    let metadata = canonical_item_agent_metadata(item);
    let status = agent_item_status_for_payload(item.status, &item.payload);
    let mut detail = serde_json::Map::from_iter([
        ("id".to_string(), json!(item.item_id.as_str())),
        ("item_id".to_string(), json!(item.item_id.as_str())),
        ("session_id".to_string(), json!(item.session_id.as_str())),
        ("thread_id".to_string(), json!(item.thread_id.as_str())),
        ("turn_id".to_string(), json!(item.turn_id.as_str())),
        ("sequence".to_string(), json!(item.sequence)),
        ("ordinal".to_string(), json!(item.ordinal)),
        ("type".to_string(), json!(item_type)),
        ("status".to_string(), json!(status)),
        (
            "started_at".to_string(),
            json!(timestamp_from_millis(item.created_at_ms)),
        ),
        (
            "updated_at".to_string(),
            json!(timestamp_from_millis(item.updated_at_ms)),
        ),
        ("metadata".to_string(), metadata),
    ]);
    if let Some(completed_at_ms) = item.completed_at_ms {
        detail.insert(
            "completed_at".to_string(),
            json!(timestamp_from_millis(completed_at_ms)),
        );
    }
    detail.extend(payload);
    if matches!(item.payload, ThreadItemPayload::Command { .. }) {
        if let Some(command_id) = item
            .metadata
            .get("source_call_id")
            .and_then(serde_json::Value::as_str)
        {
            detail.insert("command_id".to_string(), json!(command_id));
        }
    }
    serde_json::Value::Object(detail)
}

fn canonical_item_agent_metadata(item: &ThreadItem) -> serde_json::Value {
    let ThreadItemPayload::Plan {
        revision_id,
        source,
        plan,
        explanation,
        tool_call_id,
        source_item_id,
        ..
    } = &item.payload
    else {
        return item.metadata.clone();
    };
    let mut metadata = item.metadata.as_object().cloned().unwrap_or_default();
    metadata.insert("revisionId".to_string(), json!(revision_id));
    metadata.insert("plan".to_string(), json!(plan));
    if let Some(source) = source {
        metadata.insert("source".to_string(), json!(source));
    }
    if let Some(explanation) = explanation {
        metadata.insert("explanation".to_string(), json!(explanation));
    }
    if let Some(tool_call_id) = tool_call_id {
        metadata.insert("tool_call_id".to_string(), json!(tool_call_id));
    }
    if let Some(source_item_id) = source_item_id {
        metadata.insert("source_item_id".to_string(), json!(source_item_id));
    }
    serde_json::Value::Object(metadata)
}

fn canonical_payload_to_agent_detail(
    payload: &ThreadItemPayload,
) -> (&'static str, serde_json::Map<String, serde_json::Value>) {
    let mut detail = serde_json::Map::new();
    let item_type = match payload {
        ThreadItemPayload::UserMessage { content, client_id } => {
            detail.insert("content".to_string(), json!(content));
            detail.insert(
                "text".to_string(),
                json!(super::super::turn_start::user_input_text(content)),
            );
            if let Some(client_id) = client_id {
                detail.insert("client_id".to_string(), json!(client_id));
            }
            "user_message"
        }
        ThreadItemPayload::AgentMessage {
            text,
            phase,
            content_parts,
        } => {
            detail.insert("text".to_string(), json!(text));
            if let Some(phase) = phase {
                detail.insert("phase".to_string(), json!(phase));
            }
            if !content_parts.is_empty() {
                detail.insert("contentParts".to_string(), json!(content_parts));
            }
            "agent_message"
        }
        ThreadItemPayload::Plan { text, .. } => {
            detail.insert("text".to_string(), json!(text));
            "plan"
        }
        ThreadItemPayload::Reasoning { summary, content } => {
            detail.insert("text".to_string(), json!(content.join("\n")));
            detail.insert("summary".to_string(), json!(summary));
            "reasoning"
        }
        ThreadItemPayload::Tool {
            call_id,
            name,
            arguments,
            output,
        } => {
            detail.insert("call_id".to_string(), json!(call_id));
            detail.insert("tool_name".to_string(), json!(name));
            detail.insert("arguments".to_string(), json!(arguments));
            let is_web_search = is_canonical_web_search_tool(name);
            if is_web_search {
                let query = canonical_tool_argument(arguments, "query")
                    .map(str::to_string)
                    .or_else(|| canonical_web_search_action_field(arguments, "query"))
                    .unwrap_or_default();
                let action = canonical_web_search_action_field(arguments, "type")
                    .or_else(|| canonical_tool_argument(arguments, "action").map(str::to_string))
                    .filter(|value| !value.trim_start().starts_with('{'))
                    .unwrap_or_else(|| "search_query".to_string());
                detail.insert("query".to_string(), json!(query));
                detail.insert("action".to_string(), json!(action));
            }
            if let Some(output) = output {
                insert_tool_output(&mut detail, output);
            }
            if is_web_search {
                "web_search"
            } else {
                "tool_call"
            }
        }
        ThreadItemPayload::McpToolCall {
            call_id,
            server_name,
            tool_name,
            arguments,
            output,
        } => {
            detail.insert("call_id".to_string(), json!(call_id));
            detail.insert("mcp_server".to_string(), json!(server_name));
            detail.insert("tool_name".to_string(), json!(tool_name));
            detail.insert("arguments".to_string(), json!(arguments));
            if let Some(output) = output {
                insert_tool_output(&mut detail, output);
            }
            "tool_call"
        }
        ThreadItemPayload::CollabAgentToolCall {
            call_id,
            operation,
            target_thread_id,
            message,
            output,
        } => {
            detail.insert("call_id".to_string(), json!(call_id));
            if let Some(output) = output {
                insert_tool_output(&mut detail, output);
            }
            if matches!(operation, agent_protocol::CollabAgentOperation::Wait) {
                detail.insert("tool_name".to_string(), json!("wait_agent"));
                detail.insert("arguments".to_string(), json!([]));
                "tool_call"
            } else {
                detail.insert("status_label".to_string(), json!(operation));
                if let Some(target_thread_id) = target_thread_id {
                    detail.insert("session_id".to_string(), json!(target_thread_id.as_str()));
                }
                if let Some(message) = message {
                    detail.insert("summary".to_string(), json!(message));
                }
                "subagent_activity"
            }
        }
        ThreadItemPayload::Approval {
            request_id,
            action,
            scope,
            available_decisions,
            decision,
            requested_at_ms,
            resolved_at_ms,
            reason_code,
            expires_at_ms,
        } => {
            detail.insert("request_id".to_string(), json!(request_id));
            detail.insert("action_type".to_string(), json!(action.kind));
            detail.insert("prompt".to_string(), json!(action.description));
            detail.insert("scope".to_string(), json!(scope));
            detail.insert(
                "available_decisions".to_string(),
                json!(available_decisions),
            );
            if let Some(response) =
                approval::read_response(*decision, *scope, reason_code.as_deref())
            {
                detail.insert("response".to_string(), response);
            }
            if let Some(requested_at_ms) = requested_at_ms {
                detail.insert("requested_at_ms".to_string(), json!(requested_at_ms));
            }
            if let Some(resolved_at_ms) = resolved_at_ms {
                detail.insert("resolved_at_ms".to_string(), json!(resolved_at_ms));
            }
            if let Some(expires_at_ms) = expires_at_ms {
                detail.insert("expires_at_ms".to_string(), json!(expires_at_ms));
            }
            "approval_request"
        }
        ThreadItemPayload::Command {
            command,
            cwd,
            output,
            exit_code,
        } => {
            detail.insert("command".to_string(), json!(command));
            detail.insert("cwd".to_string(), json!(cwd.as_deref().unwrap_or_default()));
            if let Some(output) = output {
                detail.insert("aggregated_output".to_string(), json!(output));
            }
            if let Some(exit_code) = exit_code {
                detail.insert("exit_code".to_string(), json!(exit_code));
            }
            "command_execution"
        }
        ThreadItemPayload::File { changes, status } => {
            detail.insert("changes".to_string(), json!(changes));
            detail.insert(
                "text".to_string(),
                json!(serde_json::to_string(changes).unwrap_or_default()),
            );
            detail.insert(
                "paths".to_string(),
                json!(changes
                    .iter()
                    .map(|change| &change.path)
                    .collect::<Vec<_>>()),
            );
            detail.insert(
                "success".to_string(),
                json!(matches!(status, FileChangeStatus::Applied)),
            );
            detail.insert("source".to_string(), json!("canonical_thread_store"));
            detail.insert("file_status".to_string(), json!(status));
            "patch"
        }
        ThreadItemPayload::Media {
            uri,
            mime_type,
            preview,
        } => {
            detail.insert("uri".to_string(), json!(uri));
            detail.insert("mime_type".to_string(), json!(mime_type));
            if let Some(preview) = preview {
                detail.insert("preview".to_string(), json!(preview));
            }
            "media"
        }
        ThreadItemPayload::SubAgent {
            child_thread_id,
            activity,
            detail: activity_detail,
        } => {
            detail.insert("session_id".to_string(), json!(child_thread_id.as_str()));
            detail.insert("status_label".to_string(), json!(activity));
            if let Some(activity_detail) = activity_detail {
                detail.insert("summary".to_string(), json!(activity_detail));
            }
            "subagent_activity"
        }
        ThreadItemPayload::ContextCompaction {
            summary, window_id, ..
        } => {
            detail.insert("stage".to_string(), json!("completed"));
            if let Some(summary) = summary {
                detail.insert("detail".to_string(), json!(summary));
            }
            if let Some(window_id) = window_id {
                detail.insert("window_id".to_string(), json!(window_id));
            }
            "context_compaction"
        }
        ThreadItemPayload::Extension { name, data } => {
            detail.insert("extension_name".to_string(), json!(name));
            detail.insert("extension_data".to_string(), data.clone());
            "extension"
        }
    };
    (item_type, detail)
}

fn insert_tool_output(
    detail: &mut serde_json::Map<String, serde_json::Value>,
    output: &agent_protocol::ToolOutput,
) {
    if let Some(text) = output.text.as_ref() {
        detail.insert("output".to_string(), json!(text));
    }
    if let Some(structured_content) = output.structured_content.as_ref() {
        detail.insert("structured_content".to_string(), structured_content.clone());
        detail.insert("structuredContent".to_string(), structured_content.clone());
    }
    if let Some(error) = output.error.as_ref() {
        detail.insert("error".to_string(), json!(error));
    }
    if let Some(duration_ms) = output.duration_ms {
        detail.insert("duration_ms".to_string(), json!(duration_ms));
    }
    detail.insert("truncated".to_string(), json!(output.truncated));
    if let Some(output_ref) = output.output_ref.as_ref() {
        detail.insert("output_ref".to_string(), json!(output_ref));
    }
}

fn is_canonical_web_search_tool(name: &str) -> bool {
    matches!(
        name.trim(),
        "web_search" | "webSearch" | "search_query" | "WebSearch" | "WebSearchTool"
    )
}

fn canonical_tool_argument<'a>(
    arguments: &'a [agent_protocol::ToolArgument],
    name: &str,
) -> Option<&'a str> {
    arguments
        .iter()
        .find(|argument| argument.name == name)
        .map(|argument| argument.value.as_str())
}

fn canonical_web_search_action_field(
    arguments: &[agent_protocol::ToolArgument],
    field: &str,
) -> Option<String> {
    let action = canonical_tool_argument(arguments, "action")?;
    let action = serde_json::from_str::<serde_json::Value>(action).ok()?;
    action
        .get(field)
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn agent_item_status(status: ItemStatus) -> &'static str {
    match status {
        ItemStatus::Pending | ItemStatus::InProgress => "in_progress",
        ItemStatus::Completed => "completed",
        ItemStatus::Failed | ItemStatus::Interrupted | ItemStatus::Cancelled => "failed",
    }
}

fn agent_item_status_for_payload(
    item_status: ItemStatus,
    payload: &ThreadItemPayload,
) -> &'static str {
    match payload {
        ThreadItemPayload::File {
            status: FileChangeStatus::Proposed,
            ..
        } => "in_progress",
        ThreadItemPayload::File {
            status: FileChangeStatus::Applied,
            ..
        } => "completed",
        ThreadItemPayload::File {
            status: FileChangeStatus::Rejected | FileChangeStatus::Failed,
            ..
        } => "failed",
        _ => agent_item_status(item_status),
    }
}

fn timestamp_from_millis(timestamp_ms: i64) -> String {
    chrono::DateTime::from_timestamp_millis(timestamp_ms)
        .unwrap_or(chrono::DateTime::UNIX_EPOCH)
        .to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}
