use app_server_protocol::ProtocolKind;
use serde_json::{json, Map, Value};

pub(crate) fn body_for_model(
    model_id: &str,
    inputs: &[String],
    dimensions: Option<u32>,
    encoding_format: Option<&str>,
) -> Result<Value, String> {
    if inputs.is_empty() || inputs.iter().all(|input| input.trim().is_empty()) {
        return Err(format!(
            "{:?} requires at least one non-empty input",
            ProtocolKind::OpenaiEmbeddings
        ));
    }

    let mut body = Map::new();
    body.insert("model".to_string(), json!(model_id.trim()));
    body.insert("input".to_string(), json!(inputs));
    if let Some(dimensions) = dimensions.filter(|value| *value > 0) {
        body.insert("dimensions".to_string(), json!(dimensions));
    }
    if let Some(encoding_format) = encoding_format
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        body.insert("encoding_format".to_string(), json!(encoding_format));
    }
    Ok(Value::Object(body))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowers_batch_embedding_request() {
        let body = body_for_model(
            "text-embedding-3-small",
            &["one".to_string(), "two".to_string()],
            Some(256),
            Some("float"),
        )
        .expect("embedding body");
        assert_eq!(body["model"], "text-embedding-3-small");
        assert_eq!(body["input"], json!(["one", "two"]));
        assert_eq!(body["dimensions"], 256);
        assert_eq!(body["encoding_format"], "float");
    }
}
