use serde_json::{Map, Value};

pub(super) fn payload_source(payload: &Value) -> &Map<String, Value> {
    payload
        .get("item")
        .and_then(Value::as_object)
        .or_else(|| payload.get("data").and_then(Value::as_object))
        .unwrap_or_else(|| payload.as_object().unwrap_or_else(|| empty_object()))
}

pub(super) fn approval_payload_source(payload: &Value) -> Map<String, Value> {
    let Some(top_level) = payload.as_object() else {
        return Map::new();
    };
    let runtime_event = top_level.get("runtimeEvent").and_then(Value::as_object);
    let mut source = runtime_event
        .and_then(|event| event.get("data"))
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if let Some(data) = top_level.get("data").and_then(Value::as_object) {
        source.extend(data.clone());
    }
    if let Some(runtime_event) = runtime_event {
        source.extend(
            runtime_event
                .iter()
                .filter(|(key, _)| !matches!(key.as_str(), "type" | "data"))
                .map(|(key, value)| (key.clone(), value.clone())),
        );
    }
    source.extend(
        top_level
            .iter()
            .filter(|(key, _)| {
                !matches!(key.as_str(), "data" | "runtimeEvent")
                    && !(key.as_str() == "request_id"
                        && runtime_event.is_some_and(|event| event.contains_key("request_id")))
            })
            .map(|(key, value)| (key.clone(), value.clone())),
    );
    source
}

pub(super) fn explicit_item_id(payload: &Map<String, Value>) -> Option<String> {
    map_string(
        payload,
        &[
            "itemId",
            "item_id",
            "messageId",
            "message_id",
            "toolCallId",
            "tool_call_id",
            "toolId",
            "tool_id",
            "commandId",
            "command_id",
            "patchId",
            "patch_id",
            "actionId",
            "action_id",
            "artifactId",
            "artifact_id",
            "id",
        ],
    )
}

fn empty_object() -> &'static Map<String, Value> {
    static EMPTY: std::sync::OnceLock<Map<String, Value>> = std::sync::OnceLock::new();
    EMPTY.get_or_init(Map::new)
}

pub(super) fn value_string(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        value.get(*key).and_then(|candidate| match candidate {
            Value::String(text) if !text.trim().is_empty() => Some(text.clone()),
            Value::Number(number) => Some(number.to_string()),
            Value::Bool(boolean) => Some(boolean.to_string()),
            _ => None,
        })
    })
}

pub(super) fn value_u64(value: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|key| {
        value.get(*key).and_then(|candidate| match candidate {
            Value::Number(number) => number.as_u64(),
            Value::String(text) => text.parse().ok(),
            _ => None,
        })
    })
}

pub(super) fn value_i64(value: &Value, keys: &[&str]) -> Option<i64> {
    keys.iter().find_map(|key| {
        value.get(*key).and_then(|candidate| match candidate {
            Value::Number(number) => number.as_i64(),
            Value::String(text) => text.parse().ok(),
            _ => None,
        })
    })
}

pub(super) fn map_string(value: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        value.get(*key).and_then(|candidate| match candidate {
            Value::String(text) if !text.trim().is_empty() => Some(text.clone()),
            Value::Number(number) => Some(number.to_string()),
            Value::Bool(boolean) => Some(boolean.to_string()),
            _ => None,
        })
    })
}

pub(super) fn map_u64(value: &Map<String, Value>, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|key| {
        value.get(*key).and_then(|candidate| match candidate {
            Value::Number(number) => number.as_u64(),
            Value::String(text) => text.parse().ok(),
            _ => None,
        })
    })
}

pub(super) fn map_bool(value: &Map<String, Value>, keys: &[&str]) -> Option<bool> {
    keys.iter().find_map(|key| {
        value.get(*key).and_then(|candidate| match candidate {
            Value::Bool(value) => Some(*value),
            Value::String(value) => value.parse().ok(),
            _ => None,
        })
    })
}

pub(super) fn map_i64(value: &Map<String, Value>, keys: &[&str]) -> Option<i64> {
    keys.iter().find_map(|key| {
        value.get(*key).and_then(|candidate| match candidate {
            Value::Number(number) => number.as_i64(),
            Value::String(text) => text.parse().ok(),
            _ => None,
        })
    })
}

pub(super) fn non_empty(value: &str) -> Option<&str> {
    (!value.trim().is_empty()).then_some(value)
}

pub(super) fn message_text(payload: &Map<String, Value>) -> String {
    map_string(payload, &["text", "delta", "message", "content"])
        .or_else(|| payload.get("content").and_then(content_text))
        .or_else(|| payload.get("deltas").and_then(content_text))
        .or_else(|| {
            payload.get("input").and_then(|input| {
                value_string(input, &["text", "message", "content"])
                    .or_else(|| input.get("content").and_then(content_text))
            })
        })
        .unwrap_or_default()
}

fn content_text(content: &Value) -> Option<String> {
    match content {
        Value::Object(_) => value_string(content, &["text", "message", "content"]),
        Value::Array(parts) => {
            let text = parts
                .iter()
                .filter_map(|part| {
                    part.as_str()
                        .map(str::to_string)
                        .or_else(|| value_string(part, &["text", "message", "content"]))
                })
                .collect::<Vec<_>>()
                .join("");
            (!text.is_empty()).then_some(text)
        }
        Value::String(text) if !text.is_empty() => Some(text.clone()),
        _ => None,
    }
}

pub(super) fn string_list(payload: &Map<String, Value>, keys: &[&str]) -> Vec<String> {
    keys.iter()
        .find_map(|key| payload.get(*key))
        .map(|value| match value {
            Value::Array(values) => values
                .iter()
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect(),
            Value::String(value) => vec![value.clone()],
            _ => Vec::new(),
        })
        .unwrap_or_default()
}

pub(super) fn compact_value(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| value.to_string())
}
