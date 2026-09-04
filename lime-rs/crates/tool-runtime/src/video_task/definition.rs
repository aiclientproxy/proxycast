use crate::tool_definition::RuntimeToolDefinition;
use serde_json::{json, Map};

pub const VIDEO_TASK_TOOL_NAME: &str = "video_generate";

pub fn video_task_tool_definition() -> RuntimeToolDefinition {
    let mut properties = Map::new();
    for key in [
        "project_root_path",
        "prompt",
        "title",
        "raw_text",
        "aspect_ratio",
        "resolution",
        "image_url",
        "end_image_url",
        "provider_id",
        "model",
        "session_id",
        "thread_id",
        "turn_id",
        "project_id",
        "content_id",
        "entry_source",
        "modality_contract_key",
        "modality",
        "routing_slot",
        "requested_target",
    ] {
        properties.insert(key.to_string(), json!({ "type": "string" }));
    }
    properties.insert("runtime_contract".to_string(), json!({}));
    properties.insert(
        "duration".to_string(),
        json!({ "type": "integer", "minimum": 1 }),
    );
    properties.insert("seed".to_string(), json!({ "type": "integer" }));
    properties.insert("generate_audio".to_string(), json!({ "type": "boolean" }));
    properties.insert("camera_fixed".to_string(), json!({ "type": "boolean" }));

    RuntimeToolDefinition::new(
        VIDEO_TASK_TOOL_NAME,
        "Create a real video generation task through the App Server media task owner.",
        json!({
            "type": "object",
            "properties": properties,
            "required": ["prompt"]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definition_uses_short_current_name_and_requires_prompt() {
        let definition = video_task_tool_definition();

        assert_eq!(definition.name, VIDEO_TASK_TOOL_NAME);
        assert_eq!(definition.name, "video_generate");
        assert_eq!(definition.input_schema["required"], json!(["prompt"]));
        assert_eq!(
            definition.input_schema["properties"]["generate_audio"]["type"],
            json!("boolean")
        );
    }
}
