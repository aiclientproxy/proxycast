use serde_json::Value;

pub(super) const MAX_DISPLAY_JSON_BYTES: usize = 256 * 1024;
const MAX_DISPLAY_JSON_DEPTH: usize = 16;
const MAX_DISPLAY_COLLECTION_ITEMS: usize = 256;
pub(super) const MAX_DISPLAY_STRING_BYTES: usize = 64 * 1024;

pub(super) fn bounded_safe_json(value: Value) -> (Value, bool) {
    let mut remaining = MAX_DISPLAY_JSON_BYTES;
    let mut truncated = false;
    let value = sanitize_json_value(value, &mut remaining, 0, &mut truncated);
    if serde_json::to_vec(&value)
        .map(|bytes| bytes.len() <= MAX_DISPLAY_JSON_BYTES)
        .unwrap_or(false)
    {
        (value, truncated)
    } else {
        (serde_json::json!({ "truncated": true }), true)
    }
}

pub(super) fn bounded_safe_text(value: &str, max_bytes: usize) -> (String, bool) {
    if contains_sensitive_marker(value) {
        return (
            truncate_utf8("[redacted sensitive content]", max_bytes),
            true,
        );
    }
    if value.len() <= max_bytes {
        return (value.to_string(), false);
    }

    let suffix = truncate_utf8("... [truncated]", max_bytes);
    let prefix_limit = max_bytes.saturating_sub(suffix.len());
    let prefix = truncate_utf8(value, prefix_limit);
    (format!("{prefix}{suffix}"), true)
}

fn sanitize_json_value(
    value: Value,
    remaining: &mut usize,
    depth: usize,
    truncated: &mut bool,
) -> Value {
    if depth >= MAX_DISPLAY_JSON_DEPTH || *remaining == 0 {
        *truncated = true;
        return Value::String("[truncated]".to_string());
    }
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => value,
        Value::String(value) => {
            let limit = (*remaining).min(MAX_DISPLAY_STRING_BYTES);
            let (value, was_truncated) = bounded_safe_text(&value, limit);
            *remaining = remaining.saturating_sub(value.len());
            *truncated |= was_truncated;
            Value::String(value)
        }
        Value::Array(values) => {
            let original_len = values.len();
            let output = values
                .into_iter()
                .take(MAX_DISPLAY_COLLECTION_ITEMS)
                .map(|value| sanitize_json_value(value, remaining, depth + 1, truncated))
                .collect::<Vec<_>>();
            if original_len > output.len() {
                *truncated = true;
            }
            Value::Array(output)
        }
        Value::Object(values) => {
            let mut output = serde_json::Map::new();
            let original_len = values.len();
            let named_sensitive_value = values
                .get("name")
                .and_then(Value::as_str)
                .is_some_and(is_sensitive_json_key);
            for (key, value) in values.into_iter().take(MAX_DISPLAY_COLLECTION_ITEMS) {
                if is_sensitive_json_key(&key) || (named_sensitive_value && key == "value") {
                    output.insert(key, Value::String("[redacted]".to_string()));
                } else {
                    output.insert(
                        key,
                        sanitize_json_value(value, remaining, depth + 1, truncated),
                    );
                }
            }
            if original_len > output.len() {
                *truncated = true;
            }
            Value::Object(output)
        }
    }
}

fn truncate_utf8(value: &str, max_bytes: usize) -> String {
    let mut end = max_bytes.min(value.len());
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].to_string()
}

fn is_sensitive_json_key(key: &str) -> bool {
    let key = key.to_ascii_lowercase().replace(['-', '_'], "");
    [
        "authorization",
        "cookie",
        "credential",
        "password",
        "secret",
        "token",
        "apikey",
    ]
    .iter()
    .any(|marker| key.contains(marker))
}

fn contains_sensitive_marker(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    [
        "authorization:",
        "password=",
        "secret=",
        "token=",
        "api_key=",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_text_never_exceeds_the_requested_utf8_size() {
        for max_bytes in 0..32 {
            let (value, truncated) = bounded_safe_text("token=secret-value", max_bytes);
            assert!(truncated);
            assert!(value.len() <= max_bytes);
        }

        let long_unicode = "秘密".repeat(64);
        for max_bytes in 0..64 {
            let (value, truncated) = bounded_safe_text(&long_unicode, max_bytes);
            assert!(truncated);
            assert!(value.len() <= max_bytes);
        }
    }

    #[test]
    fn exact_collection_limit_is_not_marked_truncated() {
        let value = Value::Array(vec![Value::Null; MAX_DISPLAY_COLLECTION_ITEMS]);
        let (value, truncated) = bounded_safe_json(value);
        assert!(!truncated);
        assert_eq!(
            value.as_array().map(Vec::len),
            Some(MAX_DISPLAY_COLLECTION_ITEMS)
        );
    }
}
