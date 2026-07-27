use super::CurrentProviderError;
use runtime_core::{CanonicalLlmEvent as LlmEvent, ToolResultValue};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashSet};

const WEB_SEARCH_NAME: &str = "web_search";
const IMAGE_GENERATION_NAME: &str = "image_generation";

#[derive(Debug, Default)]
pub(super) struct HostedToolState {
    started_web_searches: HashSet<String>,
    completed_web_searches: HashSet<String>,
    started_image_generations: HashSet<String>,
    completed_image_generations: HashSet<String>,
}

pub(super) fn hosted_tool_events(
    item: &Value,
    state: &mut HostedToolState,
    terminal_event: bool,
) -> Result<Vec<LlmEvent>, CurrentProviderError> {
    match item.get("type").and_then(Value::as_str) {
        Some("web_search_call") => Ok(web_search_events(item, state, terminal_event)),
        Some("image_generation_call") => image_generation_events(item, state, terminal_event),
        _ => Ok(Vec::new()),
    }
}

fn web_search_events(
    item: &Value,
    state: &mut HostedToolState,
    terminal_event: bool,
) -> Vec<LlmEvent> {
    let id = response_item_id(item, WEB_SEARCH_NAME);
    let action = item
        .get("action")
        .filter(|value| value.is_object())
        .cloned()
        .unwrap_or_else(|| json!({}));
    let terminal = is_terminal(item, terminal_event);
    if action.as_object().is_some_and(Map::is_empty) && !terminal {
        return Vec::new();
    }

    lifecycle_events(
        item,
        id,
        WEB_SEARCH_NAME,
        action,
        terminal,
        &mut state.started_web_searches,
        &mut state.completed_web_searches,
    )
}

fn image_generation_events(
    item: &Value,
    state: &mut HostedToolState,
    terminal_event: bool,
) -> Result<Vec<LlmEvent>, CurrentProviderError> {
    let id = response_item_id(item, IMAGE_GENERATION_NAME);
    let terminal = is_terminal(item, terminal_event);
    if terminal
        && requires_image_result(item)
        && item.get("result").and_then(Value::as_str).is_none()
    {
        return Err(CurrentProviderError::new(
            "Responses image_generation_call completed without a string result",
        ));
    }
    let input = item
        .get("revised_prompt")
        .and_then(Value::as_str)
        .map(|prompt| json!({ "revised_prompt": prompt }))
        .unwrap_or_else(|| json!({}));

    Ok(lifecycle_events(
        item,
        id,
        IMAGE_GENERATION_NAME,
        input,
        terminal,
        &mut state.started_image_generations,
        &mut state.completed_image_generations,
    ))
}

fn lifecycle_events(
    item: &Value,
    id: String,
    name: &str,
    input: Value,
    terminal: bool,
    started: &mut HashSet<String>,
    completed: &mut HashSet<String>,
) -> Vec<LlmEvent> {
    let mut events = Vec::new();
    if started.insert(id.clone()) {
        events.push(LlmEvent::ToolInputStart {
            id: id.clone(),
            name: name.to_string(),
        });
        events.push(LlmEvent::ToolInputDelta {
            id: id.clone(),
            name: name.to_string(),
            text: serde_json::to_string(&input).unwrap_or_else(|_| "{}".to_string()),
        });
        events.push(LlmEvent::ToolInputEnd {
            id: id.clone(),
            name: name.to_string(),
        });
        events.push(LlmEvent::ToolCall {
            id: id.clone(),
            name: name.to_string(),
            input,
            provider_executed: Some(true),
            provider_metadata: BTreeMap::from([("raw_response_item".to_string(), item.clone())]),
        });
    }
    if terminal && completed.insert(id.clone()) {
        events.push(LlmEvent::ToolResult {
            id,
            name: name.to_string(),
            result: ToolResultValue::Json {
                value: item.clone(),
            },
            provider_executed: Some(true),
        });
    }
    events
}

fn response_item_id(item: &Value, fallback_prefix: &str) -> String {
    item.get("id")
        .or_else(|| item.get("item_id"))
        .or_else(|| item.get("output_index"))
        .map(|value| match value {
            Value::String(value) => value.clone(),
            other => other.to_string(),
        })
        .unwrap_or_else(|| format!("{fallback_prefix}-0"))
}

fn is_terminal(item: &Value, terminal_event: bool) -> bool {
    terminal_event
        || item
            .get("status")
            .and_then(Value::as_str)
            .is_some_and(|status| matches!(status, "completed" | "failed" | "cancelled"))
}

fn requires_image_result(item: &Value) -> bool {
    !item
        .get("status")
        .and_then(Value::as_str)
        .is_some_and(|status| matches!(status, "failed" | "cancelled"))
}
