use app_server_protocol::AgentEvent;
use serde_json::{json, Value};
use std::collections::HashSet;

pub(super) fn file_artifact_items_from_events(events: &[AgentEvent]) -> Vec<Value> {
    let mut seen = HashSet::new();
    let mut items = Vec::new();
    for event in events.iter().rev() {
        let Some(item) = file_artifact_item_from_event(event) else {
            continue;
        };
        let Some(id) = item.get("id").and_then(Value::as_str) else {
            continue;
        };
        if seen.insert(id.to_string()) {
            items.push(item);
        }
    }
    items.sort_by(|left, right| {
        right
            .get("updated_at")
            .and_then(Value::as_str)
            .cmp(&left.get("updated_at").and_then(Value::as_str))
    });
    items
}

fn file_artifact_item_from_event(event: &AgentEvent) -> Option<Value> {
    if event.event_type != "file.changed" {
        return None;
    }
    let path = string_field(&event.payload, &["path", "filePath", "file_path"])?;
    let artifact_id = string_field(&event.payload, &["artifactId", "artifact_id"])
        .or_else(|| string_array_field(&event.payload, &["artifactRefs", "artifact_refs"]).pop())
        .unwrap_or_else(|| stable_scope_id("artifact:file", path.as_str()));
    let checkpoint_id = string_field(
        &event.payload,
        &[
            "checkpointRef",
            "checkpoint_ref",
            "checkpointId",
            "checkpoint_id",
        ],
    )
    .unwrap_or_else(|| {
        stable_scope_id(
            "checkpoint:file",
            format!("{}:{}", event.event_id, path).as_str(),
        )
    });
    let preview = string_field(
        &event.payload,
        &["preview", "summary", "previewText", "preview_text"],
    );
    let mut metadata = serde_json::Map::new();
    metadata.insert("artifactId".to_string(), json!(artifact_id));
    metadata.insert("artifactRequestId".to_string(), json!(event.event_id));
    metadata.insert("artifactVersionId".to_string(), json!(checkpoint_id));
    metadata.insert("artifactVersionNo".to_string(), json!(event.sequence));
    metadata.insert("artifactKind".to_string(), json!("code_file"));
    metadata.insert("artifactStatus".to_string(), json!("ready"));
    insert_string_metadata(
        &mut metadata,
        "eventClass",
        &event.payload,
        &["eventClass", "event_class"],
    );
    insert_string_metadata(
        &mut metadata,
        "operation",
        &event.payload,
        &["operation", "action"],
    );
    insert_string_metadata(
        &mut metadata,
        "toolName",
        &event.payload,
        &["toolName", "tool_name"],
    );
    metadata.insert(
        "previewText".to_string(),
        json!(preview.clone().unwrap_or_else(|| path.clone())),
    );
    metadata.insert("checkpointRef".to_string(), json!(checkpoint_id));
    insert_string_metadata(
        &mut metadata,
        "contentRef",
        &event.payload,
        &["contentRef", "content_ref"],
    );
    insert_string_metadata(
        &mut metadata,
        "diffRef",
        &event.payload,
        &["diffRef", "diff_ref"],
    );
    insert_string_metadata(
        &mut metadata,
        "checkpointSnapshotFile",
        &event.payload,
        &["checkpointSnapshotFile", "checkpoint_snapshot_file"],
    );
    if let Some(change) = file_change_from_payload(&event.payload) {
        metadata.insert("file_change".to_string(), change);
    }
    if let Some(diff) = event.payload.get("diff").cloned() {
        metadata.insert("artifactVersionDiff".to_string(), diff);
    } else if let Some(diff_ref) = string_field(&event.payload, &["diffRef", "diff_ref"]) {
        metadata.insert(
            "artifactVersionDiff".to_string(),
            json!({ "diffRef": diff_ref }),
        );
    }
    metadata.insert(
        "artifactVersion".to_string(),
        json!({
            "id": checkpoint_id,
            "versionNo": event.sequence,
            "snapshotPath": string_field(&event.payload, &["snapshotPath", "snapshot_path"]).unwrap_or_else(|| path.clone()),
            "title": preview.clone().unwrap_or_else(|| path.clone()),
            "kind": "code_file",
            "status": "ready",
        }),
    );
    let content = previous_content_from_payload(&event.payload);

    Some(compact_object(json!({
        "id": checkpoint_id,
        "type": "file_artifact",
        "thread_id": event.thread_id,
        "turn_id": event.turn_id,
        "path": path,
        "source": "runtime",
        "status": "completed",
        "content": content,
        "metadata": metadata,
        "started_at": event.timestamp,
        "completed_at": event.timestamp,
        "updated_at": event.timestamp,
    })))
}

fn insert_string_metadata(
    metadata: &mut serde_json::Map<String, Value>,
    key: &str,
    payload: &Value,
    keys: &[&str],
) {
    if let Some(value) = string_field(payload, keys) {
        metadata.insert(key.to_string(), json!(value));
    }
}

fn file_change_from_payload(payload: &Value) -> Option<Value> {
    let mut change = payload.get("change").cloned().unwrap_or_else(|| json!({}));
    let Value::Object(ref mut change_object) = change else {
        return Some(change);
    };
    if !change_object.contains_key("previousContent") {
        if let Some(previous_content) = previous_content_from_payload(payload) {
            change_object.insert("previousContent".to_string(), json!(previous_content));
        }
    }
    if !change_object.contains_key("previousContentSnapshotFile") {
        if let Some(snapshot_file) = string_field(
            payload,
            &["checkpointSnapshotFile", "checkpoint_snapshot_file"],
        ) {
            change_object.insert(
                "previousContentSnapshotFile".to_string(),
                json!(snapshot_file),
            );
        }
    }
    if change_object.is_empty() {
        None
    } else {
        Some(change)
    }
}

fn previous_content_from_payload(payload: &Value) -> Option<String> {
    string_field(
        payload,
        &[
            "content",
            "previousContent",
            "previous_content",
            "beforeContent",
            "before_content",
            "oldContent",
            "old_content",
        ],
    )
    .or_else(|| {
        let change = payload.get("change")?;
        string_field(
            change,
            &[
                "previousContent",
                "previous_content",
                "beforeContent",
                "before_content",
                "oldContent",
                "old_content",
            ],
        )
    })
}

fn string_field(value: &Value, keys: &[&str]) -> Option<String> {
    let object = value.as_object()?;
    keys.iter()
        .find_map(|key| object.get(*key))
        .and_then(value_string)
}

fn string_array_field(value: &Value, keys: &[&str]) -> Vec<String> {
    let Some(object) = value.as_object() else {
        return Vec::new();
    };
    keys.iter()
        .filter_map(|key| object.get(*key))
        .flat_map(value_string_vec)
        .collect()
}

fn value_string_vec(value: &Value) -> Vec<String> {
    if let Some(values) = value.as_array() {
        return values.iter().filter_map(value_string).collect();
    }
    value_string(value).into_iter().collect()
}

fn value_string(value: &Value) -> Option<String> {
    value
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
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
