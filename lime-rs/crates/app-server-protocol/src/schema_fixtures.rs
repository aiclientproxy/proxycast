use crate::app_server_method_catalog;
use crate::generated_schema_tree;
use crate::AppServerMethodKind;
use crate::AppServerRequestSerializationScope;
use crate::JsonRpcError;
use crate::JsonRpcErrorResponse;
use crate::JsonRpcMessage;
use crate::JsonRpcNotification;
use crate::JsonRpcRequest;
use crate::JsonRpcResponse;
use crate::RequestId;
use crate::APP_SERVER_REQUEST_SERIALIZATION_SCOPES;
use crate::PROTOCOL_VERSION;
use serde_json::json;
use serde_json::Map;
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

pub fn protocol_fixture_manifest() -> Value {
    let methods = app_server_method_catalog();
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        "jsonRpc": {
            "version": crate::JSONRPC_VERSION,
            "envelopes": [
                "request",
                "notification",
                "response",
                "error"
            ],
            "sendsJsonRpcVersionField": false
        },
        "methods": methods
            .iter()
            .map(|spec| {
                json!({
                    "method": spec.method,
                    "kind": match spec.kind {
                        AppServerMethodKind::Request => "request",
                        AppServerMethodKind::Notification => "notification",
                        AppServerMethodKind::ServerRequest => "serverRequest",
                    }
                })
            })
            .collect::<Vec<_>>(),
        "requestSerializationScopes": APP_SERVER_REQUEST_SERIALIZATION_SCOPES
            .iter()
            .map(|spec| {
                json!({
                    "method": spec.method,
                    "scope": request_serialization_scope_name(spec.scope),
                })
            })
            .collect::<Vec<_>>(),
        "schemas": {
            "jsonrpc": crate::JSONRPC_SCHEMA_TYPE_NAMES,
            "v0": crate::V0_SCHEMA_TYPE_NAMES,
            "v2": crate::V2_SCHEMA_TYPE_NAMES
        }
    })
}

fn request_serialization_scope_name(scope: AppServerRequestSerializationScope) -> &'static str {
    match scope {
        AppServerRequestSerializationScope::Thread => "thread",
        AppServerRequestSerializationScope::McpOauth => "mcpOauth",
        AppServerRequestSerializationScope::McpResourceSubscription => "mcpResourceSubscription",
    }
}

pub fn generated_fixture_tree() -> BTreeMap<PathBuf, Vec<u8>> {
    let mut fixtures = BTreeMap::new();
    for (path, bytes) in generated_schema_tree() {
        let bytes = normalize_fixture_bytes(&path, &bytes).expect("normalize generated schema");
        fixtures.insert(path, bytes);
    }
    fixtures.insert(
        PathBuf::from("json/manifest.json"),
        stable_json_fixture_bytes(&protocol_fixture_manifest()),
    );
    fixtures.insert(
        PathBuf::from("json/envelopes/request.json"),
        stable_json_fixture_bytes(
            &serde_json::to_value(JsonRpcMessage::Request(JsonRpcRequest::new(
                RequestId::Integer(1),
                crate::METHOD_INITIALIZE,
                Some(json!({ "clientInfo": { "name": "fixture" } })),
            )))
            .expect("request fixture"),
        ),
    );
    fixtures.insert(
        PathBuf::from("json/envelopes/notification.json"),
        stable_json_fixture_bytes(
            &serde_json::to_value(JsonRpcMessage::Notification(JsonRpcNotification::new(
                crate::METHOD_INITIALIZED,
                Some(json!({})),
            )))
            .expect("notification fixture"),
        ),
    );
    fixtures.insert(
        PathBuf::from("json/envelopes/response.json"),
        stable_json_fixture_bytes(
            &serde_json::to_value(JsonRpcMessage::Response(
                JsonRpcResponse::new(RequestId::Integer(1), json!({ "ok": true }))
                    .expect("response"),
            ))
            .expect("response fixture"),
        ),
    );
    fixtures.insert(
        PathBuf::from("json/envelopes/error.json"),
        stable_json_fixture_bytes(
            &serde_json::to_value(JsonRpcMessage::Error(JsonRpcErrorResponse {
                id: RequestId::Integer(1),
                error: JsonRpcError::new(crate::error_codes::RUNTIME_ERROR, "runtime error"),
            }))
            .expect("error fixture"),
        ),
    );
    fixtures
}

pub fn read_fixture_tree(root: &Path) -> Result<BTreeMap<PathBuf, Vec<u8>>, String> {
    collect_files(root, root)
}

pub fn write_fixture_tree(root: &Path) -> Result<(), String> {
    let fixtures = generated_fixture_tree();
    if root.exists() {
        fs::remove_dir_all(root)
            .map_err(|error| format!("failed to remove {}: {error}", root.display()))?;
    }
    fs::create_dir_all(root)
        .map_err(|error| format!("failed to create {}: {error}", root.display()))?;
    for (relative_path, content) in fixtures {
        let path = root.join(relative_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
        }
        fs::write(&path, content)
            .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
    }
    Ok(())
}

pub fn assert_fixture_trees_match(
    label: &str,
    fixture_tree: &BTreeMap<PathBuf, Vec<u8>>,
    generated_tree: &BTreeMap<PathBuf, Vec<u8>>,
) -> Result<(), String> {
    let fixture_paths = fixture_tree.keys().collect::<Vec<_>>();
    let generated_paths = generated_tree.keys().collect::<Vec<_>>();
    if fixture_paths != generated_paths {
        return Err(format!(
            "{label} fixture file set does not match generated output: expected {:?}, got {:?}",
            fixture_paths, generated_paths
        ));
    }

    for (path, expected) in fixture_tree {
        let actual = generated_tree
            .get(path)
            .ok_or_else(|| format!("missing generated fixture {}", path.display()))?;
        if expected != actual {
            return Err(format!(
                "{label} fixture {} differs from generated output",
                path.display()
            ));
        }
    }
    Ok(())
}

pub fn normalize_fixture_bytes(path: &Path, bytes: &[u8]) -> Result<Vec<u8>, String> {
    if path
        .extension()
        .is_some_and(|extension| extension == "json")
    {
        let value: Value = serde_json::from_slice(bytes)
            .map_err(|error| format!("failed to parse JSON in {}: {error}", path.display()))?;
        return Ok(pretty_json_bytes(&canonicalize_json(&value)));
    }
    Ok(bytes.to_vec())
}

fn collect_files(root: &Path, current: &Path) -> Result<BTreeMap<PathBuf, Vec<u8>>, String> {
    let mut files = BTreeMap::new();
    if !current.exists() {
        return Ok(files);
    }

    for entry in fs::read_dir(current)
        .map_err(|error| format!("failed to read {}: {error}", current.display()))?
    {
        let entry = entry
            .map_err(|error| format!("failed to read {} entry: {error}", current.display()))?;
        let path = entry.path();
        if path.is_dir() {
            files.extend(collect_files(root, &path)?);
            continue;
        }
        if path.is_file() {
            let relative_path = path
                .strip_prefix(root)
                .map_err(|error| format!("failed to relativize {}: {error}", path.display()))?
                .to_path_buf();
            let bytes = fs::read(&path)
                .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
            files.insert(
                relative_path.clone(),
                normalize_fixture_bytes(&relative_path, &bytes)?,
            );
        }
    }

    Ok(files)
}

fn pretty_json_bytes(value: &Value) -> Vec<u8> {
    let mut bytes = serde_json::to_vec_pretty(value).expect("serialize fixture JSON");
    bytes.push(b'\n');
    bytes
}

fn stable_json_fixture_bytes(value: &Value) -> Vec<u8> {
    pretty_json_bytes(&canonicalize_json(value))
}

fn canonicalize_json(value: &Value) -> Value {
    match value {
        Value::Array(items) => {
            let items = items.iter().map(canonicalize_json).collect::<Vec<_>>();
            let mut sortable = Vec::with_capacity(items.len());
            for item in &items {
                let Some(key) = schema_array_item_sort_key(item) else {
                    return Value::Array(items);
                };
                let stable = serde_json::to_string(item).unwrap_or_default();
                sortable.push((key, stable));
            }

            let mut items = items.into_iter().zip(sortable).collect::<Vec<_>>();
            items.sort_by(
                |(_, (left_key, left_stable)), (_, (right_key, right_stable))| match left_key
                    .cmp(right_key)
                {
                    Ordering::Equal => left_stable.cmp(right_stable),
                    other => other,
                },
            );
            Value::Array(items.into_iter().map(|(item, _)| item).collect())
        }
        Value::Object(map) => {
            let mut entries = map.iter().collect::<Vec<_>>();
            entries.sort_by_key(|(key, _)| *key);
            let mut sorted = Map::with_capacity(map.len());
            for (key, child) in entries {
                sorted.insert(key.clone(), canonicalize_json(child));
            }
            Value::Object(sorted)
        }
        _ => value.clone(),
    }
}

fn schema_array_item_sort_key(item: &Value) -> Option<String> {
    match item {
        Value::Null => Some("null".to_string()),
        Value::Bool(value) => Some(format!("b:{value}")),
        Value::Number(value) => Some(format!("n:{value}")),
        Value::String(value) => Some(format!("s:{value}")),
        Value::Object(map) => map
            .get("method")
            .and_then(Value::as_str)
            .map(|method| format!("method:{method}")),
        Value::Array(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::v2::SERVER_REQUEST_METHODS;

    #[test]
    fn protocol_fixture_manifest_lists_current_methods_without_jsonrpc_field() {
        let manifest = protocol_fixture_manifest();

        assert_eq!(manifest["protocolVersion"], PROTOCOL_VERSION);
        assert_eq!(manifest["jsonRpc"]["sendsJsonRpcVersionField"], false);
        assert_eq!(
            manifest["methods"]
                .as_array()
                .expect("methods")
                .iter()
                .filter(|method| method["kind"] == "request")
                .count(),
            app_server_method_catalog()
                .iter()
                .filter(|method| method.kind == AppServerMethodKind::Request)
                .count()
        );
        let methods = manifest["methods"].as_array().expect("methods");
        for method in SERVER_REQUEST_METHODS {
            assert!(methods
                .iter()
                .any(|spec| { spec["method"] == *method && spec["kind"] == "serverRequest" }));
        }
        assert_eq!(
            manifest["requestSerializationScopes"]
                .as_array()
                .expect("requestSerializationScopes")
                .len(),
            APP_SERVER_REQUEST_SERIALIZATION_SCOPES.len()
        );
    }

    #[test]
    fn generated_fixture_tree_is_stable_and_canonical_json() {
        let fixtures = generated_fixture_tree();

        assert!(fixtures.contains_key(Path::new("json/app_server_protocol.schemas.json")));
        assert!(fixtures.contains_key(Path::new("json/manifest.json")));
        assert!(fixtures.contains_key(Path::new("json/envelopes/request.json")));
        for (path, bytes) in fixtures {
            let normalized = normalize_fixture_bytes(&path, &bytes).expect("normalize");
            assert_eq!(normalized, bytes);
        }
    }

    #[test]
    fn fixture_tree_comparison_reports_path_and_content_drift() {
        let generated = generated_fixture_tree();
        let mut missing = generated.clone();
        missing.remove(Path::new("json/manifest.json"));
        assert!(assert_fixture_trees_match("json", &generated, &missing).is_err());

        let mut changed = generated.clone();
        changed.insert(PathBuf::from("json/manifest.json"), b"{}\n".to_vec());
        assert!(assert_fixture_trees_match("json", &generated, &changed).is_err());
        assert!(assert_fixture_trees_match("json", &generated, &generated).is_ok());
    }
}
