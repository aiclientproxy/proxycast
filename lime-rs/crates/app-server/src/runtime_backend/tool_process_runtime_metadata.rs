use super::tool_process_metadata::{
    build_tool_process_metadata, merge_result_tool_process_metadata, merge_tool_process_metadata,
    promote_mcp_route_metadata, SoulStyleMetadata, ToolProcessMetadataInput, ToolProcessStatus,
};
use agent_protocol::{ItemStatus, ThreadItem, ThreadItemPayload, ToolArgument, ToolOutput};
use lime_agent::{AgentEvent as RuntimeAgentEvent, AgentToolResult};
use serde_json::{Map, Value};
use std::collections::HashMap;

pub(super) fn enrich_runtime_tool_process_payload(
    event: &RuntimeAgentEvent,
    payload_object: &mut Map<String, Value>,
    fallback_soul_style: Option<&SoulStyleMetadata>,
) {
    if matches!(event, RuntimeAgentEvent::ToolProgress { .. }) {
        promote_mcp_route_metadata(payload_object);
    }
    let Some(input) = runtime_tool_process_input(event) else {
        return;
    };
    let payload_soul_style = SoulStyleMetadata::from_payload_object(payload_object)
        .unwrap_or_default()
        .merge_with_fallback(fallback_soul_style);
    let metadata = build_tool_process_metadata(ToolProcessMetadataInput {
        tool_id: &input.tool_id,
        tool_name: input.tool_name.as_deref(),
        status: input.status,
        arguments: input.arguments.as_ref(),
        result: input.result.as_ref(),
        soul_style: payload_soul_style.as_ref(),
    });
    if let Some(item_metadata) = canonical_item_metadata_mut(payload_object) {
        merge_canonical_item_process_metadata(item_metadata, &metadata);
        insert_runtime_tool_process_aliases(item_metadata, &metadata);
    } else {
        merge_tool_process_metadata(payload_object, &metadata);
        merge_result_tool_process_metadata(payload_object, &metadata);
        insert_runtime_tool_process_aliases(payload_object, &metadata);
    }
}

struct RuntimeToolProcessInput {
    tool_id: String,
    tool_name: Option<String>,
    status: ToolProcessStatus,
    arguments: Option<Value>,
    result: Option<AgentToolResult>,
}

fn runtime_tool_process_input(event: &RuntimeAgentEvent) -> Option<RuntimeToolProcessInput> {
    match event {
        RuntimeAgentEvent::ItemStarted { item } => {
            canonical_tool_process_input(item, ToolProcessStatus::Started)
        }
        RuntimeAgentEvent::ItemUpdated { item } => {
            canonical_tool_process_input(item, ToolProcessStatus::Progress)
        }
        RuntimeAgentEvent::ItemCompleted { item } => canonical_tool_process_input(
            item,
            if item.status == ItemStatus::Completed {
                ToolProcessStatus::Completed
            } else {
                ToolProcessStatus::Failed
            },
        ),
        RuntimeAgentEvent::ToolProgress { tool_id, progress } => Some(RuntimeToolProcessInput {
            tool_id: tool_id.clone(),
            tool_name: progress
                .metadata
                .as_ref()
                .and_then(|metadata| read_metadata_string(metadata, &["tool_name", "toolName"])),
            status: ToolProcessStatus::Progress,
            arguments: None,
            result: None,
        }),
        RuntimeAgentEvent::ToolOutputDelta {
            tool_id, metadata, ..
        } => Some(RuntimeToolProcessInput {
            tool_id: tool_id.clone(),
            tool_name: metadata
                .as_ref()
                .and_then(|metadata| read_metadata_string(metadata, &["tool_name", "toolName"])),
            status: ToolProcessStatus::OutputDelta,
            arguments: None,
            result: None,
        }),
        RuntimeAgentEvent::ToolInputDelta {
            tool_id,
            tool_name,
            accumulated_arguments,
            ..
        } => Some(RuntimeToolProcessInput {
            tool_id: tool_id.clone(),
            tool_name: tool_name.clone(),
            status: ToolProcessStatus::InputDelta,
            arguments: accumulated_arguments
                .as_deref()
                .and_then(non_empty_str)
                .map(parse_tool_arguments),
            result: None,
        }),
        _ => None,
    }
}

fn canonical_tool_process_input(
    item: &ThreadItem,
    status: ToolProcessStatus,
) -> Option<RuntimeToolProcessInput> {
    let ThreadItemPayload::Tool {
        call_id,
        name,
        arguments,
        output,
    } = &item.payload
    else {
        return None;
    };
    Some(RuntimeToolProcessInput {
        tool_id: call_id.clone(),
        tool_name: Some(name.clone()),
        status,
        arguments: canonical_arguments_value(arguments),
        result: output
            .as_ref()
            .map(|output| canonical_agent_tool_result(item, output)),
    })
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

fn canonical_arguments_value(arguments: &[ToolArgument]) -> Option<Value> {
    if arguments.is_empty() {
        return None;
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

fn canonical_item_metadata_mut(
    payload_object: &mut Map<String, Value>,
) -> Option<&mut Map<String, Value>> {
    let item = payload_object.get_mut("item")?.as_object_mut()?;
    let metadata = item
        .entry("metadata".to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    metadata.as_object_mut()
}

fn merge_canonical_item_process_metadata(
    target: &mut Map<String, Value>,
    metadata: &Map<String, Value>,
) {
    for (key, value) in metadata {
        if key == "tool_process_summary" && target.contains_key(key) {
            continue;
        }
        if matches!(key.as_str(), "tool_process_facts" | "soul_lifecycle") {
            if let (Some(target), Some(source)) = (
                target.get_mut(key).and_then(Value::as_object_mut),
                value.as_object(),
            ) {
                for (field, value) in source {
                    target.entry(field.clone()).or_insert_with(|| value.clone());
                }
                continue;
            }
        }
        target.entry(key.clone()).or_insert_with(|| value.clone());
    }
}

fn insert_runtime_tool_process_aliases(
    payload_object: &mut Map<String, Value>,
    metadata: &Map<String, Value>,
) {
    if let Some(facts) = metadata.get("tool_process_facts") {
        payload_object.insert("toolProcessFacts".to_string(), facts.clone());
    }
    if let Some(summary) = metadata.get("tool_process_summary") {
        payload_object.insert("toolProcessSummary".to_string(), summary.clone());
    }
}

fn parse_tool_arguments(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| Value::String(arguments.to_string()))
}

fn read_metadata_string(metadata: &HashMap<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| metadata.get(*key)?.as_str().and_then(non_empty_str))
        .map(str::to_string)
}

fn non_empty_str(value: &str) -> Option<&str> {
    let value = value.trim();
    (!value.is_empty()).then_some(value)
}
