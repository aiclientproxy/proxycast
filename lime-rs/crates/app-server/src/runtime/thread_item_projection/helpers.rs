use crate::runtime::{string_field, StoredSession};
use app_server_protocol::AgentEvent;
use serde_json::{json, Map, Value};

pub(super) fn sort_thread_items(items: &mut [Value]) {
    items.sort_by(|left, right| {
        let left_sequence = left
            .get("sequence")
            .and_then(Value::as_u64)
            .unwrap_or(u64::MAX);
        let right_sequence = right
            .get("sequence")
            .and_then(Value::as_u64)
            .unwrap_or(u64::MAX);
        left_sequence
            .cmp(&right_sequence)
            .then_with(|| item_timestamp(left).cmp(&item_timestamp(right)))
            .then_with(|| item_id(left).cmp(&item_id(right)))
    });
}

fn item_timestamp(item: &Value) -> String {
    string_field(
        item,
        &["started_at", "updated_at", "completed_at", "created_at"],
    )
    .unwrap_or_default()
}

fn item_id(item: &Value) -> String {
    string_field(item, &["id"]).unwrap_or_default()
}

pub(super) fn lifecycle_base_item(
    stored: &StoredSession,
    event: &AgentEvent,
    id: &str,
    item_type: &str,
    status: &str,
) -> Value {
    let mut object = serde_json::Map::new();
    object.insert("id".to_string(), Value::String(id.to_string()));
    object.insert(
        "thread_id".to_string(),
        Value::String(
            event
                .thread_id
                .clone()
                .unwrap_or_else(|| stored.session.thread_id.clone()),
        ),
    );
    object.insert(
        "turn_id".to_string(),
        Value::String(event.turn_id.clone().unwrap_or_default()),
    );
    object.insert("sequence".to_string(), json!(event.sequence));
    object.insert("type".to_string(), Value::String(item_type.to_string()));
    object.insert("status".to_string(), Value::String(status.to_string()));
    object.insert(
        "started_at".to_string(),
        Value::String(event.timestamp.clone()),
    );
    object.insert(
        "updated_at".to_string(),
        Value::String(event.timestamp.clone()),
    );
    if status != "in_progress" {
        object.insert(
            "completed_at".to_string(),
            Value::String(event.timestamp.clone()),
        );
    }
    Value::Object(object)
}

pub(super) fn update_lifecycle_item(
    object: &mut Map<String, Value>,
    event: &AgentEvent,
    status: &str,
) {
    object.insert("status".to_string(), json!(status));
    object.insert("updated_at".to_string(), json!(event.timestamp));
    object
        .entry("turn_id".to_string())
        .or_insert_with(|| json!(event.turn_id));
    if status != "in_progress" {
        object.insert("completed_at".to_string(), json!(event.timestamp));
    }
    if let Some(existing_sequence) = object.get("sequence").and_then(Value::as_u64) {
        if event.sequence < existing_sequence {
            object.insert("sequence".to_string(), json!(event.sequence));
        }
    }
}

pub(super) fn merge_optional_field(
    object: &mut Map<String, Value>,
    key: &str,
    value: Option<Value>,
) {
    let Some(value) = value else {
        return;
    };
    if value.is_null() {
        return;
    }
    if matches!(&value, Value::Array(items) if items.is_empty()) {
        return;
    }
    if value.as_str().is_some_and(str::is_empty) {
        return;
    }
    object.insert(key.to_string(), value);
}

pub(super) fn merge_lifecycle_metadata(object: &mut Map<String, Value>, event: &AgentEvent) {
    let metadata = object
        .entry("metadata".to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    let Some(metadata) = metadata.as_object_mut() else {
        return;
    };
    if let Some(source_metadata) = event.payload.get("metadata").and_then(Value::as_object) {
        for (key, value) in source_metadata {
            metadata.insert(key.clone(), value.clone());
        }
    }
    merge_metadata_array(
        metadata,
        "source_event_ids",
        Value::String(event.event_id.clone()),
    );
    merge_metadata_array(
        metadata,
        "source_event_types",
        Value::String(event.event_type.clone()),
    );
    metadata.insert("source_event_id".to_string(), json!(event.event_id));
    metadata.insert("source_event_type".to_string(), json!(event.event_type));
    if let Some(value) = event.payload.get("sourceClient").cloned() {
        metadata.insert("source_client".to_string(), value);
    }
    if let Some(value) = event.payload.get("sourceProvenance").cloned() {
        metadata.insert("source_provenance".to_string(), value);
    }
    if let Some(value) = event.payload.get("imported").cloned() {
        metadata.insert("imported".to_string(), value);
    }
    if let Some(value) = event.payload.get("importedSynthetic").cloned() {
        metadata.insert("imported_synthetic".to_string(), value);
    }
    if let Some(value) = event.payload.get("importedIncomplete").cloned() {
        metadata.insert("imported_incomplete".to_string(), value);
    }
}

fn merge_metadata_array(metadata: &mut Map<String, Value>, key: &str, value: Value) {
    let entry = metadata
        .entry(key.to_string())
        .or_insert_with(|| Value::Array(Vec::new()));
    let Some(values) = entry.as_array_mut() else {
        return;
    };
    if !values.iter().any(|existing| existing == &value) {
        values.push(value);
    }
}

pub(super) fn base_item(
    stored: &StoredSession,
    event: &AgentEvent,
    item_type: &str,
    status: &str,
    fields: Value,
) -> Value {
    let mut object = serde_json::Map::new();
    object.insert(
        "id".to_string(),
        Value::String(format!("{}:{}", event.event_type, event.event_id)),
    );
    object.insert(
        "thread_id".to_string(),
        Value::String(
            event
                .thread_id
                .clone()
                .unwrap_or_else(|| stored.session.thread_id.clone()),
        ),
    );
    object.insert(
        "turn_id".to_string(),
        Value::String(event.turn_id.clone().unwrap_or_default()),
    );
    object.insert("sequence".to_string(), json!(event.sequence));
    object.insert("type".to_string(), Value::String(item_type.to_string()));
    object.insert("status".to_string(), Value::String(status.to_string()));
    object.insert(
        "started_at".to_string(),
        Value::String(event.timestamp.clone()),
    );
    object.insert(
        "updated_at".to_string(),
        Value::String(event.timestamp.clone()),
    );
    if status != "in_progress" {
        object.insert(
            "completed_at".to_string(),
            Value::String(event.timestamp.clone()),
        );
    }
    merge_object_fields(&mut object, fields);
    Value::Object(object)
}

fn merge_object_fields(target: &mut serde_json::Map<String, Value>, fields: Value) {
    let Value::Object(fields) = fields else {
        return;
    };
    for (key, value) in fields {
        if value.is_null() {
            continue;
        }
        if matches!(&value, Value::Array(items) if items.is_empty()) {
            continue;
        }
        target.insert(key, value);
    }
}

pub(super) fn compact_json(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .filter_map(|(key, value)| {
                    if value.is_null() {
                        return None;
                    }
                    if matches!(&value, Value::Array(items) if items.is_empty()) {
                        return None;
                    }
                    Some((key, compact_json(value)))
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.into_iter().map(compact_json).collect()),
        value => value,
    }
}

pub(super) fn event_metadata(event: &AgentEvent) -> Value {
    let mut metadata = serde_json::Map::new();
    metadata.insert("source_event_id".to_string(), json!(event.event_id));
    metadata.insert("source_event_type".to_string(), json!(event.event_type));
    if let Some(value) = event.payload.get("sourceClient").cloned() {
        metadata.insert("source_client".to_string(), value);
    }
    if let Some(value) = event.payload.get("sourceProvenance").cloned() {
        metadata.insert("source_provenance".to_string(), value);
    }
    if let Some(value) = event.payload.get("imported").cloned() {
        metadata.insert("imported".to_string(), value);
    }
    if let Some(value) = event.payload.get("importedSynthetic").cloned() {
        metadata.insert("imported_synthetic".to_string(), value);
    }
    if let Some(value) = event.payload.get("importedIncomplete").cloned() {
        metadata.insert("imported_incomplete".to_string(), value);
    }
    if let Some(value) = event.payload.get("importedReadOnly").cloned() {
        metadata.insert("imported_read_only".to_string(), value);
    }
    if let Some(value) = event.payload.get("outputRefs").cloned() {
        metadata.insert("output_refs".to_string(), value);
    }
    Value::Object(metadata)
}
