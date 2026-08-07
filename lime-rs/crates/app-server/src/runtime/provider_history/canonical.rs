use super::{flush_assistant, flush_tool_results, PROVIDER_TOOL_OUTPUT_MAX_BYTES};
use agent_protocol::{AgentInput, ItemStatus, ThreadItem, ThreadItemPayload, ToolArgument};
use agent_runtime::reply_input::RuntimeReplyInput;
use app_server_protocol::AgentEvent;
use model_provider::current_client::{
    CurrentProviderContent, CurrentProviderMessage, CurrentProviderToolCall,
    CurrentProviderToolResult,
};
use serde_json::Value;
use std::collections::HashMap;

pub(super) fn append_fork_item<G>(
    item: &ThreadItem,
    messages: &mut Vec<CurrentProviderMessage>,
    assistant_content: &mut Vec<CurrentProviderContent>,
    assistant_text_by_item: &mut HashMap<String, String>,
    tool_results: &mut Vec<CurrentProviderContent>,
    provider_input: &mut G,
) -> Result<(), String>
where
    G: FnMut(Vec<AgentInput>) -> Result<RuntimeReplyInput, String>,
{
    match &item.payload {
        ThreadItemPayload::UserMessage { content, .. } => {
            flush_assistant(messages, assistant_content, assistant_text_by_item);
            flush_tool_results(messages, tool_results);
            let input = provider_input(content.clone())?;
            if let Some(message) = super::user_message_from_input(&input) {
                messages.push(message);
            }
        }
        ThreadItemPayload::AgentMessage {
            text,
            content_parts,
            ..
        } => {
            if content_parts
                .iter()
                .any(|part| matches!(part, agent_protocol::MessageContentPart::Media { .. }))
            {
                return Err(format!(
                    "fork canonical agent message {} contains unsupported media",
                    item.item_id
                ));
            }
            flush_tool_results(messages, tool_results);
            let text = if text.trim().is_empty() {
                content_parts
                    .iter()
                    .filter_map(|part| match part {
                        agent_protocol::MessageContentPart::Text { text } => Some(text.as_str()),
                        agent_protocol::MessageContentPart::Media { .. } => None,
                    })
                    .collect::<String>()
            } else {
                text.clone()
            };
            if !text.is_empty() {
                assistant_content.push(CurrentProviderContent::Text(text));
            }
        }
        ThreadItemPayload::Reasoning { summary, content } => {
            flush_tool_results(messages, tool_results);
            let reasoning = if content.is_empty() {
                summary.concat()
            } else {
                content.concat()
            };
            if !reasoning.is_empty() {
                assistant_content.push(CurrentProviderContent::Reasoning(reasoning));
            }
        }
        ThreadItemPayload::Tool { .. }
        | ThreadItemPayload::DynamicToolCall { .. }
        | ThreadItemPayload::McpToolCall { .. } => {
            let call = tool_call_from_item(item).ok_or_else(|| {
                format!(
                    "fork canonical tool item {} omitted a valid call",
                    item.item_id
                )
            })?;
            let result = tool_result_from_item(item).ok_or_else(|| {
                format!(
                    "fork canonical tool item {} omitted a valid result",
                    item.item_id
                )
            })?;
            flush_tool_results(messages, tool_results);
            assistant_content.push(CurrentProviderContent::ToolCall(call));
            flush_assistant(messages, assistant_content, assistant_text_by_item);
            tool_results.push(CurrentProviderContent::ToolResult(result));
        }
        ThreadItemPayload::CollabAgentToolCall { .. }
        | ThreadItemPayload::Media { .. }
        | ThreadItemPayload::ContextCompaction { .. }
        | ThreadItemPayload::Hook { .. }
        | ThreadItemPayload::Unknown { .. }
        | ThreadItemPayload::Extension { .. } => {
            return Err(format!(
                "fork canonical item {} cannot be lowered without loss",
                item.item_id
            ));
        }
        ThreadItemPayload::Plan { .. }
        | ThreadItemPayload::Approval { .. }
        | ThreadItemPayload::Command { .. }
        | ThreadItemPayload::File { .. }
        | ThreadItemPayload::SubAgent { .. } => {}
    }
    Ok(())
}

pub(super) fn tool_call_from_event(event: &AgentEvent) -> Option<CurrentProviderToolCall> {
    let item = canonical_tool_item(event)?;
    tool_call_from_item(&item)
}

fn tool_call_from_item(item: &ThreadItem) -> Option<CurrentProviderToolCall> {
    let (call_id, name, arguments, raw_arguments) = match &item.payload {
        ThreadItemPayload::Tool {
            call_id,
            name,
            arguments,
            ..
        } => {
            let (arguments, raw_arguments) = canonical_tool_arguments(arguments);
            (call_id.clone(), name.clone(), arguments, raw_arguments)
        }
        ThreadItemPayload::DynamicToolCall {
            call_id,
            namespace,
            tool,
            arguments,
            ..
        } => {
            let name = namespace
                .as_ref()
                .map(|namespace| format!("{namespace}__{tool}"))
                .unwrap_or_else(|| tool.clone());
            let raw_arguments = serde_json::to_string(arguments).ok()?;
            (call_id.clone(), name, arguments.clone(), raw_arguments)
        }
        ThreadItemPayload::McpToolCall {
            call_id,
            server_name,
            tool_name,
            arguments,
            ..
        } => {
            let inner_name =
                lime_mcp::naming::extract_runtime_inner_tool_name(server_name, tool_name)
                    .unwrap_or(tool_name);
            let (arguments, raw_arguments) = canonical_tool_arguments(arguments);
            (
                call_id.clone(),
                lime_mcp::naming::build_runtime_tool_name(server_name, inner_name),
                arguments,
                raw_arguments,
            )
        }
        _ => return None,
    };
    Some(CurrentProviderToolCall {
        id: call_id,
        name,
        arguments,
        raw_arguments,
        provider_metadata: item
            .metadata
            .get("provider_metadata")
            .and_then(Value::as_object)
            .map(|metadata| {
                metadata
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect()
            })
            .unwrap_or_default(),
    })
}

pub(super) fn tool_result_from_event(event: &AgentEvent) -> Option<CurrentProviderToolResult> {
    let item = canonical_tool_item(event)?;
    tool_result_from_item(&item)
}

fn tool_result_from_item(item: &ThreadItem) -> Option<CurrentProviderToolResult> {
    let status = item.status;
    if let ThreadItemPayload::DynamicToolCall {
        call_id,
        namespace,
        tool,
        content_items,
        success,
        ..
    } = &item.payload
    {
        let name = namespace
            .as_ref()
            .map(|namespace| format!("{namespace}__{tool}"))
            .unwrap_or_else(|| tool.clone());
        let success = success.unwrap_or(status == ItemStatus::Completed);
        let error = (!success).then(|| {
            content_items
                .iter()
                .find_map(|item| match item {
                    agent_protocol::DynamicToolCallContentItem::InputText { text }
                        if !text.trim().is_empty() =>
                    {
                        Some(text.trim().to_string())
                    }
                    _ => None,
                })
                .unwrap_or_else(|| "dynamic tool call failed".to_string())
        });
        return Some(CurrentProviderToolResult {
            call_id: call_id.clone(),
            name,
            success,
            output: serde_json::to_string(content_items).ok()?,
            error,
        });
    }
    let (call_id, name, output) = match &item.payload {
        ThreadItemPayload::Tool {
            call_id,
            name,
            output,
            ..
        } => (call_id.clone(), name.clone(), output.as_ref()),
        ThreadItemPayload::McpToolCall {
            call_id,
            server_name,
            tool_name,
            output,
            ..
        } => {
            let inner_name =
                lime_mcp::naming::extract_runtime_inner_tool_name(server_name, tool_name)
                    .unwrap_or(tool_name);
            (
                call_id.clone(),
                lime_mcp::naming::build_runtime_tool_name(server_name, inner_name),
                output.as_ref(),
            )
        }
        _ => return None,
    };
    let output = output?;
    let output_ref = output.output_ref.as_deref();
    let output_text = output
        .text
        .as_deref()
        .filter(|text| !text.trim().is_empty())
        .map(ToString::to_string)
        .or_else(|| output.structured_content.as_ref().map(Value::to_string))
        .map(|text| {
            tool_runtime::tool_io::format_tool_output_for_model(
                &text,
                tool_runtime::tool_io::ToolOutputTruncationPolicy::Bytes(
                    PROVIDER_TOOL_OUTPUT_MAX_BYTES,
                ),
            )
        })
        .or_else(|| {
            output_ref.map(|reference| {
                format!(
                    "Tool output was omitted from context; retained artifact reference: {reference}"
                )
            })
        })
        .unwrap_or_default();
    Some(CurrentProviderToolResult {
        call_id,
        name,
        success: status == ItemStatus::Completed,
        output: output_text,
        error: output
            .error
            .as_ref()
            .filter(|error| !error.trim().is_empty())
            .cloned(),
    })
}

fn canonical_tool_item(event: &AgentEvent) -> Option<ThreadItem> {
    if !matches!(event.event_type.as_str(), "item.started" | "item.completed") {
        return None;
    }
    serde_json::from_value(event.payload.get("item")?.clone()).ok()
}

fn canonical_tool_arguments(arguments: &[ToolArgument]) -> (Value, String) {
    let arguments = if let [argument] = arguments {
        if argument.name == "value" {
            serde_json::from_str(&argument.value)
                .unwrap_or_else(|_| Value::String(argument.value.clone()))
        } else {
            canonical_tool_argument_object(arguments)
        }
    } else {
        canonical_tool_argument_object(arguments)
    };
    let raw_arguments = serde_json::to_string(&arguments).unwrap_or_else(|_| "{}".to_string());
    (arguments, raw_arguments)
}

fn canonical_tool_argument_object(arguments: &[ToolArgument]) -> Value {
    Value::Object(
        arguments
            .iter()
            .map(|argument| {
                let value = serde_json::from_str(&argument.value)
                    .unwrap_or_else(|_| Value::String(argument.value.clone()));
                (argument.name.clone(), value)
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_protocol::{
        DynamicToolCallContentItem, ItemId, ItemKind, SessionId, ThreadId, ToolArgument,
        ToolOutput, TurnId,
    };

    fn mcp_item(tool_name: &str) -> ThreadItem {
        ThreadItem {
            session_id: SessionId::new("session-1"),
            thread_id: ThreadId::new("thread-1"),
            turn_id: TurnId::new("turn-1"),
            item_id: ItemId::new("mcp-item"),
            sequence: 1,
            ordinal: 1,
            created_at_ms: 1,
            updated_at_ms: 2,
            completed_at_ms: Some(2),
            kind: ItemKind::McpToolCall,
            status: ItemStatus::Completed,
            payload: ThreadItemPayload::McpToolCall {
                call_id: "mcp-call".to_string(),
                server_name: "docs".to_string(),
                tool_name: tool_name.to_string(),
                mcp_app_resource_uri: None,
                plugin_id: None,
                arguments: vec![ToolArgument {
                    name: "query".to_string(),
                    value: "\"fork\"".to_string(),
                }],
                output: Some(ToolOutput {
                    text: Some("found".to_string()),
                    ..ToolOutput::default()
                }),
            },
            metadata: Value::Null,
        }
    }

    #[test]
    fn mcp_item_uses_one_canonical_runtime_tool_name() {
        for tool_name in ["search", "mcp__docs__search"] {
            let item = mcp_item(tool_name);
            let call = tool_call_from_item(&item).expect("MCP tool call");
            let result = tool_result_from_item(&item).expect("MCP tool result");

            assert_eq!(call.name, "mcp__docs__search");
            assert_eq!(result.name, "mcp__docs__search");
            assert_eq!(call.arguments, serde_json::json!({"query": "fork"}));
            assert_eq!(result.output, "found");
        }
    }

    #[test]
    fn tool_item_restores_provider_metadata_for_history_lowering() {
        let mut item = mcp_item("search");
        item.metadata = serde_json::json!({
            "provider_metadata": {
                "google": { "thoughtSignature": "sig" }
            }
        });

        let call = tool_call_from_item(&item).expect("tool call with provider metadata");
        assert_eq!(
            call.provider_metadata
                .get("google")
                .and_then(|value| value.get("thoughtSignature")),
            Some(&serde_json::json!("sig"))
        );
    }

    #[test]
    fn dynamic_tool_item_restores_typed_call_and_ordered_result() {
        let content_items = vec![
            DynamicToolCallContentItem::InputText {
                text: "Lime".to_string(),
            },
            DynamicToolCallContentItem::InputImage {
                image_url: "data:image/png;base64,AA==".to_string(),
            },
            DynamicToolCallContentItem::InputAudio {
                audio_url: "data:audio/wav;base64,AA==".to_string(),
            },
        ];
        let item = ThreadItem {
            session_id: SessionId::new("session-1"),
            thread_id: ThreadId::new("thread-1"),
            turn_id: TurnId::new("turn-1"),
            item_id: ItemId::new("dynamic-item"),
            sequence: 1,
            ordinal: 1,
            created_at_ms: 1,
            updated_at_ms: 2,
            completed_at_ms: Some(2),
            kind: ItemKind::DynamicToolCall,
            status: ItemStatus::Completed,
            payload: ThreadItemPayload::DynamicToolCall {
                call_id: "dynamic-call".to_string(),
                namespace: Some("desktop".to_string()),
                tool: "appInfo".to_string(),
                arguments: serde_json::json!({
                    "includeLocale": true,
                    "options": {"platform": "darwin"}
                }),
                content_items: content_items.clone(),
                success: Some(true),
                duration_ms: Some(17),
            },
            metadata: Value::Null,
        };

        let call = tool_call_from_item(&item).expect("dynamic tool call");
        let result = tool_result_from_item(&item).expect("dynamic tool result");
        assert_eq!(call.id, "dynamic-call");
        assert_eq!(call.name, "desktop__appInfo");
        assert_eq!(
            call.arguments,
            serde_json::json!({
                "includeLocale": true,
                "options": {"platform": "darwin"}
            })
        );
        assert_eq!(
            serde_json::from_str::<Value>(&call.raw_arguments).expect("raw arguments"),
            call.arguments
        );
        assert_eq!(result.call_id, "dynamic-call");
        assert_eq!(result.name, "desktop__appInfo");
        assert!(result.success);
        assert_eq!(result.error, None);
        assert_eq!(
            serde_json::from_str::<Vec<DynamicToolCallContentItem>>(&result.output)
                .expect("ordered dynamic result"),
            content_items
        );
    }
}
