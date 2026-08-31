use super::tool_process_metadata::{
    build_tool_process_metadata, merge_result_tool_process_metadata, merge_tool_process_metadata,
    promote_mcp_route_metadata, SoulStyleMetadata, ToolProcessMetadataInput, ToolProcessStatus,
};
use app_server_protocol::AgentEvent;
use lime_agent::AgentToolResult;
use serde_json::{Map, Value};
use std::collections::HashMap;

pub(crate) fn enrich_external_tool_process_payload(
    existing_events: &[AgentEvent],
    event_type: &str,
    payload_object: &mut Map<String, Value>,
    fallback_soul_style: Option<&SoulStyleMetadata>,
) {
    promote_mcp_route_metadata(payload_object);
    let Some(status) = tool_process_status_from_event_type(event_type) else {
        return;
    };
    let Some(tool_id) = payload_string_from_map(
        payload_object,
        &["toolCallId", "tool_call_id", "toolId", "tool_id", "id"],
    ) else {
        return;
    };

    let active_tool = active_tool_for_call_id(existing_events, &tool_id);
    let tool_name = tool_name_from_payload(payload_object).or_else(|| {
        active_tool
            .as_ref()
            .and_then(|tool| tool.tool_name.as_deref())
            .map(str::to_string)
    });
    let arguments = tool_arguments_from_payload(payload_object).or_else(|| {
        active_tool
            .as_ref()
            .and_then(|tool| tool.arguments.as_ref())
            .cloned()
    });
    let result = matches!(
        status,
        ToolProcessStatus::Completed | ToolProcessStatus::Failed
    )
    .then(|| external_tool_result(payload_object, event_type));
    let status = match (status, result.as_ref()) {
        (ToolProcessStatus::Completed, Some(result)) if !result.success => {
            ToolProcessStatus::Failed
        }
        (status, _) => status,
    };
    let soul_style = SoulStyleMetadata::from_payload_object(payload_object)
        .unwrap_or_default()
        .merge_with_fallback(
            active_tool
                .as_ref()
                .and_then(|tool| tool.soul_style.as_ref()),
        )
        .unwrap_or_default()
        .merge_with_fallback(fallback_soul_style);
    let metadata = build_tool_process_metadata(ToolProcessMetadataInput {
        tool_id: &tool_id,
        tool_name: tool_name.as_deref(),
        status,
        arguments: arguments.as_ref(),
        result: result.as_ref(),
        soul_style: soul_style.as_ref(),
    });
    merge_tool_process_metadata(payload_object, &metadata);
    merge_result_tool_process_metadata(payload_object, &metadata);
    insert_tool_process_top_level_aliases(payload_object, &metadata);
}

pub(crate) fn enrich_external_tool_policy_payload(
    existing_events: &[AgentEvent],
    event_type: &str,
    payload_object: &mut Map<String, Value>,
    fallback_soul_style: Option<&SoulStyleMetadata>,
) {
    let Some((phase, status)) = tool_policy_lifecycle_status(event_type) else {
        return;
    };
    let Some(tool_id) = payload_string_from_map(
        payload_object,
        &["toolCallId", "tool_call_id", "toolId", "tool_id", "id"],
    ) else {
        return;
    };
    let active_tool = active_tool_for_call_id(existing_events, &tool_id);
    let tool_name = tool_name_from_payload(payload_object).or_else(|| {
        active_tool
            .as_ref()
            .and_then(|tool| tool.tool_name.as_deref())
            .map(str::to_string)
    });
    let mut lifecycle = Map::from_iter([
        (
            "surface".to_string(),
            Value::String("tool_lifecycle".to_string()),
        ),
        ("phase".to_string(), Value::String(phase.to_string())),
        ("status".to_string(), Value::String(status.to_string())),
        ("styleLevel".to_string(), Value::String("L4".to_string())),
        ("riskLevel".to_string(), Value::String("high".to_string())),
    ]);
    let mut facts = Map::from_iter([
        (
            "source".to_string(),
            Value::String("runtime_facts".to_string()),
        ),
        ("toolCallId".to_string(), Value::String(tool_id)),
        (
            "policyEvent".to_string(),
            Value::String(normalize_policy_event_type(event_type).to_string()),
        ),
        ("status".to_string(), Value::String(status.to_string())),
        ("phase".to_string(), Value::String(phase.to_string())),
        ("styleLevel".to_string(), Value::String("L4".to_string())),
        ("riskLevel".to_string(), Value::String("high".to_string())),
    ]);
    if let Some(tool_name) = tool_name.and_then(|value| non_empty_string(value.as_str())) {
        facts.insert("toolName".to_string(), Value::String(tool_name));
    }
    if let Some(action_kind) =
        payload_string_from_map(payload_object, &["actionKind", "action_type", "actionType"])
    {
        facts.insert("actionKind".to_string(), Value::String(action_kind));
    }
    if let Some(reason_code) = payload_string_from_map(
        payload_object,
        &[
            "reasonCode",
            "reason_code",
            "failureCategory",
            "failure_category",
        ],
    ) {
        facts.insert("reasonCode".to_string(), Value::String(reason_code));
    }
    let soul_style = SoulStyleMetadata::from_payload_object(payload_object)
        .unwrap_or_default()
        .merge_with_fallback(
            active_tool
                .as_ref()
                .and_then(|tool| tool.soul_style.as_ref()),
        )
        .unwrap_or_default()
        .merge_with_fallback(fallback_soul_style);
    if let Some(soul_style) = soul_style.as_ref() {
        soul_style.insert_lifecycle_fields(&mut lifecycle);
        soul_style.insert_fact_fields(&mut facts);
    }

    let mut metadata = Map::from_iter([
        (
            "tool_process_facts".to_string(),
            Value::Object(facts.clone()),
        ),
        (
            "soul_lifecycle".to_string(),
            Value::Object(lifecycle.clone()),
        ),
        (
            "soul_surface".to_string(),
            Value::String("tool_lifecycle".to_string()),
        ),
        ("soul_phase".to_string(), Value::String(phase.to_string())),
        ("style_level".to_string(), Value::String("L4".to_string())),
        ("risk_level".to_string(), Value::String("high".to_string())),
    ]);
    if let Some(soul_style) = soul_style.as_ref() {
        soul_style.insert_top_level_fields(&mut metadata);
    }
    merge_policy_tool_metadata(payload_object, &metadata);
    insert_tool_process_top_level_aliases(payload_object, &metadata);
}

fn tool_policy_lifecycle_status(event_type: &str) -> Option<(&'static str, &'static str)> {
    match normalize_policy_event_type(event_type) {
        "action.required" => Some(("tool_policy_review", "action_required")),
        "permission.denied" => Some(("after_tool_failure", "permission_denied")),
        "sandbox.blocked" => Some(("after_tool_failure", "sandbox_blocked")),
        _ => None,
    }
}

fn normalize_policy_event_type(event_type: &str) -> &str {
    match event_type {
        "action_required" => "action.required",
        "permission_denied" => "permission.denied",
        "sandbox_blocked" => "sandbox.blocked",
        value => value,
    }
}

fn merge_policy_tool_metadata(
    payload_object: &mut Map<String, Value>,
    metadata: &Map<String, Value>,
) {
    for (key, value) in metadata {
        merge_policy_metadata_field(payload_object, key, value);
    }
    let metadata_value = payload_object
        .entry("metadata".to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    if !metadata_value.is_object() {
        *metadata_value = Value::Object(Map::new());
    }
    if let Some(metadata_object) = metadata_value.as_object_mut() {
        for (key, value) in metadata {
            merge_policy_metadata_field(metadata_object, key, value);
        }
    }
}

fn merge_policy_metadata_field(target: &mut Map<String, Value>, key: &str, value: &Value) {
    if matches!(key, "tool_process_facts" | "soul_lifecycle") {
        if let (Some(target_object), Some(source_object)) = (
            target.get_mut(key).and_then(Value::as_object_mut),
            value.as_object(),
        ) {
            for (field, field_value) in source_object {
                target_object
                    .entry(field.clone())
                    .or_insert_with(|| field_value.clone());
            }
            return;
        }
    }
    target
        .entry(key.to_string())
        .or_insert_with(|| value.clone());
}

#[derive(Clone, Debug)]
struct ActiveToolProcess {
    tool_name: Option<String>,
    arguments: Option<Value>,
    soul_style: Option<SoulStyleMetadata>,
}

fn tool_process_status_from_event_type(event_type: &str) -> Option<ToolProcessStatus> {
    match normalize_tool_event_type(event_type) {
        "tool.started" => Some(ToolProcessStatus::Started),
        "tool.args" | "tool.args.delta" | "tool.input.delta" => Some(ToolProcessStatus::InputDelta),
        "tool.progress" => Some(ToolProcessStatus::Progress),
        "tool.output.delta" => Some(ToolProcessStatus::OutputDelta),
        "tool.result" => Some(ToolProcessStatus::Completed),
        "tool.failed" => Some(ToolProcessStatus::Failed),
        _ => None,
    }
}

fn normalize_tool_event_type(event_type: &str) -> &str {
    match event_type {
        "tool_args" => "tool.args",
        "tool_args_delta" => "tool.args.delta",
        "tool_input_delta" => "tool.input.delta",
        "tool_output_delta" => "tool.output.delta",
        value => value,
    }
}

fn active_tool_for_call_id(
    existing_events: &[AgentEvent],
    tool_call_id: &str,
) -> Option<ActiveToolProcess> {
    let mut active_tool = None;
    for event in existing_events {
        let Some(payload_object) = event.payload.as_object() else {
            continue;
        };
        let Some(event_tool_call_id) = payload_string_from_map(
            payload_object,
            &["toolCallId", "tool_call_id", "toolId", "tool_id", "id"],
        ) else {
            continue;
        };
        if event_tool_call_id != tool_call_id {
            continue;
        }

        match normalize_tool_event_type(&event.event_type) {
            "tool.started" => {
                active_tool = Some(ActiveToolProcess {
                    tool_name: tool_name_from_payload(payload_object),
                    arguments: tool_arguments_from_payload(payload_object),
                    soul_style: SoulStyleMetadata::from_payload_object(payload_object),
                });
            }
            "tool.args" | "tool.args.delta" | "tool.input.delta" => {
                if let Some(tool) = active_tool.as_mut() {
                    if tool.tool_name.is_none() {
                        tool.tool_name = tool_name_from_payload(payload_object);
                    }
                    if let Some(arguments) = tool_arguments_from_payload(payload_object) {
                        tool.arguments = Some(arguments);
                    }
                    if tool.soul_style.is_none() {
                        tool.soul_style = SoulStyleMetadata::from_payload_object(payload_object);
                    }
                }
            }
            "tool.result" | "tool.failed" => {
                active_tool = None;
            }
            _ => {}
        }
    }
    active_tool
}

fn insert_tool_process_top_level_aliases(
    payload_object: &mut Map<String, Value>,
    metadata: &Map<String, Value>,
) {
    if let Some(facts) = metadata.get("tool_process_facts") {
        payload_object
            .entry("toolProcessFacts".to_string())
            .or_insert_with(|| facts.clone());
    }
    if let Some(summary) = metadata.get("tool_process_summary") {
        payload_object
            .entry("toolProcessSummary".to_string())
            .or_insert_with(|| summary.clone());
    }
}

fn tool_name_from_payload(payload_object: &Map<String, Value>) -> Option<String> {
    payload_string_from_map(payload_object, &["toolName", "tool_name", "name", "tool"]).or_else(
        || {
            payload_object
                .get("result")
                .and_then(Value::as_object)
                .and_then(|result| {
                    result
                        .get("metadata")
                        .and_then(Value::as_object)
                        .and_then(|metadata| {
                            payload_string_from_map(metadata, &["toolName", "tool_name"])
                        })
                })
        },
    )
}

fn tool_arguments_from_payload(payload_object: &Map<String, Value>) -> Option<Value> {
    ["arguments", "args", "params", "parameters", "input"]
        .iter()
        .find_map(|key| payload_object.get(*key).and_then(tool_arguments_from_value))
        .or_else(|| {
            ["rawArgs", "raw_args", "delta"]
                .iter()
                .find_map(|key| payload_object.get(*key).and_then(tool_arguments_from_value))
        })
}

fn tool_arguments_from_value(value: &Value) -> Option<Value> {
    match value {
        Value::Null => None,
        Value::String(value) => normalize_external_arguments(value),
        _ => Some(value.clone()),
    }
}

fn normalize_external_arguments(value: &str) -> Option<Value> {
    let value = non_empty_str(value)?;
    serde_json::from_str(value)
        .ok()
        .or_else(|| Some(Value::String(value.to_string())))
}

fn external_tool_result(payload_object: &Map<String, Value>, event_type: &str) -> AgentToolResult {
    let result_object = payload_object.get("result").and_then(Value::as_object);
    let success = result_object
        .and_then(|result| result.get("success").and_then(Value::as_bool))
        .or_else(|| payload_object.get("success").and_then(Value::as_bool))
        .unwrap_or_else(|| normalize_tool_event_type(event_type) != "tool.failed");
    let output = result_object
        .and_then(|result| {
            payload_string_from_map(result, &["output", "content", "text", "message"])
        })
        .or_else(|| payload_string_from_map(payload_object, &["output", "content", "text"]))
        .unwrap_or_default();
    let error = result_object
        .and_then(|result| {
            payload_string_from_map(
                result,
                &[
                    "error",
                    "errorMessage",
                    "message",
                    "failureCategory",
                    "reasonCode",
                ],
            )
        })
        .or_else(|| {
            payload_string_from_map(
                payload_object,
                &[
                    "error",
                    "errorMessage",
                    "message",
                    "failureCategory",
                    "reasonCode",
                ],
            )
        });
    let structured_content = result_object
        .and_then(|result| {
            result
                .get("structuredContent")
                .or_else(|| result.get("structured_content"))
        })
        .cloned()
        .or_else(|| {
            payload_object
                .get("structuredContent")
                .or_else(|| payload_object.get("structured_content"))
                .cloned()
        });
    AgentToolResult {
        success,
        output,
        error,
        structured_content,
        images: None,
        metadata: external_result_metadata(payload_object, result_object),
    }
}

fn external_result_metadata(
    payload_object: &Map<String, Value>,
    result_object: Option<&Map<String, Value>>,
) -> Option<HashMap<String, Value>> {
    let mut metadata = HashMap::new();
    if let Some(result_metadata) = result_object
        .and_then(|result| result.get("metadata"))
        .and_then(Value::as_object)
    {
        for (key, value) in result_metadata {
            metadata.insert(key.clone(), value.clone());
        }
    }
    if let Some(payload_metadata) = payload_object.get("metadata").and_then(Value::as_object) {
        for (key, value) in payload_metadata {
            metadata.entry(key.clone()).or_insert_with(|| value.clone());
        }
    }
    for key in [
        "failureCategory",
        "failure_category",
        "reasonCode",
        "reason_code",
        "status",
    ] {
        if let Some(value) = payload_object.get(key) {
            metadata
                .entry(key.to_string())
                .or_insert_with(|| value.clone());
        }
    }
    (!metadata.is_empty()).then_some(metadata)
}

fn payload_string_from_value(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => non_empty_str(value).map(str::to_string),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn payload_string_from_map(payload_object: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| payload_object.get(*key).and_then(payload_string_from_value))
}

fn non_empty_str(value: &str) -> Option<&str> {
    let value = value.trim();
    (!value.is_empty()).then_some(value)
}

fn non_empty_string(value: &str) -> Option<String> {
    non_empty_str(value).map(str::to_string)
}
