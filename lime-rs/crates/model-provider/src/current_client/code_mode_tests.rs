use super::lowering::responses_message;
use runtime_core::{
    CanonicalMessage, CanonicalRole, ContentPart, ProviderMetadata, ToolResultValue,
};
use serde_json::json;
use std::collections::BTreeMap;

#[test]
fn custom_tool_error_keeps_formatted_runtime_output() {
    let message = CanonicalMessage {
        id: None,
        role: CanonicalRole::Tool,
        content: vec![ContentPart::CustomToolResult {
            id: "custom-call-failed".to_string(),
            name: "exec".to_string(),
            result: ToolResultValue::Error {
                value: json!({
                    "output": "Script failed\nOutput:\npartial\nScript error:\nboom",
                    "error": "boom",
                }),
            },
            error: Some("boom".to_string()),
            metadata: ProviderMetadata::new(),
        }],
        metadata: ProviderMetadata::new(),
    };

    assert_eq!(
        responses_message(&message, &BTreeMap::new()),
        vec![json!({
            "type": "custom_tool_call_output",
            "call_id": "custom-call-failed",
            "output": "Script failed\nOutput:\npartial\nScript error:\nboom",
        })]
    );
}
