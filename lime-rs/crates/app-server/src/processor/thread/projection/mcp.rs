use agent_protocol as canonical;
use app_server_protocol::protocol::v2;
use serde_json::Value;

use super::safe_display::{
    bounded_safe_json, bounded_safe_text, MAX_DISPLAY_JSON_BYTES, MAX_DISPLAY_STRING_BYTES,
};

pub(super) fn project_mcp_app_context(
    context: canonical::McpToolCallAppContext,
) -> v2::McpToolCallAppContext {
    v2::McpToolCallAppContext {
        connector_id: context.connector_id,
        link_id: context.link_id,
        resource_uri: context.resource_uri,
        app_name: context.app_name,
        template_id: None,
        action_name: context.action_name,
    }
}

pub(super) fn project_mcp_status(status: canonical::ItemStatus) -> v2::McpToolCallStatus {
    match status {
        canonical::ItemStatus::Pending | canonical::ItemStatus::InProgress => {
            v2::McpToolCallStatus::InProgress
        }
        canonical::ItemStatus::Completed => v2::McpToolCallStatus::Completed,
        canonical::ItemStatus::Failed
        | canonical::ItemStatus::Interrupted
        | canonical::ItemStatus::Cancelled => v2::McpToolCallStatus::Failed,
    }
}

pub(super) fn project_mcp_tool_result(output: &canonical::ToolOutput) -> v2::McpToolCallResult {
    let mut truncated = output.truncated;
    let content = output
        .text
        .as_ref()
        .filter(|text| !text.is_empty())
        .map(|text| {
            let (text, text_truncated) = bounded_safe_text(text, MAX_DISPLAY_STRING_BYTES);
            truncated |= text_truncated;
            vec![serde_json::json!({ "type": "text", "text": text })]
        })
        .unwrap_or_default();
    let structured_content = output.structured_content.clone().map(|value| {
        let (value, value_truncated) = bounded_safe_json(value);
        truncated |= value_truncated;
        value
    });

    let mut metadata = serde_json::Map::new();
    if truncated {
        metadata.insert("truncated".to_string(), Value::Bool(true));
    }
    if output.output_ref.is_some() {
        metadata.insert("outputAvailable".to_string(), Value::Bool(true));
    }

    let mut result = v2::McpToolCallResult {
        content,
        structured_content,
        meta: (!metadata.is_empty()).then_some(Value::Object(metadata)),
    };
    if serde_json::to_vec(&result)
        .map(|bytes| bytes.len() > MAX_DISPLAY_JSON_BYTES)
        .unwrap_or(true)
    {
        result = v2::McpToolCallResult {
            content: vec![serde_json::json!({
                "type": "text",
                "text": "[tool output exceeded display limit]"
            })],
            structured_content: None,
            meta: Some(serde_json::json!({
                "truncated": true,
                "outputAvailable": output.output_ref.is_some()
            })),
        };
    }
    result
}
