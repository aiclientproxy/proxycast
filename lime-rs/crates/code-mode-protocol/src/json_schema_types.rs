//! Minimal JSON Schema to TypeScript rendering used by tool descriptions.

use serde_json::Value;

pub fn render_json_schema_to_typescript(schema: &Value) -> String {
    render(schema, 0)
}

fn render(schema: &Value, depth: usize) -> String {
    if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
        return reference
            .rsplit('/')
            .next()
            .unwrap_or("unknown")
            .to_string();
    }
    if let Some(options) = schema.get("enum").and_then(Value::as_array) {
        return options
            .iter()
            .map(|value| match value {
                Value::String(value) => format!("{value:?}"),
                _ => value.to_string(),
            })
            .collect::<Vec<_>>()
            .join(" | ");
    }
    if let Some(options) = schema.get("anyOf").and_then(Value::as_array) {
        return options
            .iter()
            .map(|value| render(value, depth))
            .collect::<Vec<_>>()
            .join(" | ");
    }
    match schema.get("type").and_then(Value::as_str) {
        Some("string") => "string".to_string(),
        Some("number") | Some("integer") => "number".to_string(),
        Some("boolean") => "boolean".to_string(),
        Some("null") => "null".to_string(),
        Some("array") => format!(
            "{}[]",
            render(schema.get("items").unwrap_or(&Value::Null), depth)
        ),
        Some("object") => {
            let required = schema.get("required").and_then(Value::as_array);
            let properties = schema.get("properties").and_then(Value::as_object);
            let Some(properties) = properties else {
                return "Record<string, unknown>".to_string();
            };
            let indent = "  ".repeat(depth + 1);
            let close = "  ".repeat(depth);
            let fields = properties
                .iter()
                .map(|(name, value)| {
                    let optional = !required
                        .is_some_and(|items| items.iter().any(|item| item.as_str() == Some(name)));
                    format!(
                        "{indent}{name}{}: {};",
                        if optional { "?" } else { "" },
                        render(value, depth + 1)
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            format!("{{\n{fields}\n{close}}}")
        }
        _ => "unknown".to_string(),
    }
}
