use super::conversions::{duration_ns, execution_outcome, tool_definition};
use code_mode_protocol::grpc as proto;
use code_mode_protocol::{
    FunctionCallOutputContentItem, ImageDetail, RuntimeCodeModeCellId, RuntimeCodeModeResponse,
};
use std::time::Duration;

#[test]
fn execution_outcome_preserves_typed_content_and_duration() {
    let response = RuntimeCodeModeResponse::Result {
        cell_id: RuntimeCodeModeCellId::new("cell"),
        content_items: vec![
            FunctionCallOutputContentItem::InputText {
                text: "hello".to_string(),
            },
            FunctionCallOutputContentItem::InputImage {
                image_url: "data:image/png;base64,AA==".to_string(),
                detail: Some(ImageDetail::Original),
            },
            FunctionCallOutputContentItem::InputAudio {
                audio_url: "data:audio/wav;base64,AA==".to_string(),
            },
        ],
        error_text: Some("failed".to_string()),
        code_mode_host_duration: None,
    };

    let actual = execution_outcome(response, 123_456_789);
    assert_eq!(actual.cell_id, "cell");
    assert_eq!(actual.code_mode_host_duration_ns, 123_456_789);
    assert!(matches!(
        actual.content_items.as_slice(),
        [
            proto::ContentItem { item: Some(proto::content_item::Item::Text(_)) },
            proto::ContentItem { item: Some(proto::content_item::Item::Image(proto::ImageContent { detail: Some(value), .. })) },
            proto::ContentItem { item: Some(proto::content_item::Item::Audio(_)) },
        ] if *value == proto::ImageDetail::Original as i32
    ));
    assert!(matches!(
        actual.outcome,
        Some(proto::execution_outcome::Outcome::Completed(proto::ExecutionCompleted {
            error_text: Some(error)
        })) if error == "failed"
    ));
}

#[test]
fn duration_conversion_saturates_at_wire_limit() {
    assert_eq!(duration_ns(Duration::from_nanos(7)), 7);
    assert_eq!(duration_ns(Duration::new(u64::MAX, 999_999_999)), u64::MAX);
}

#[test]
fn tool_definition_requires_kind_name_and_valid_schema() {
    let base = proto::ToolDefinition {
        name: "echo".to_string(),
        tool_name: Some(proto::ToolName {
            name: "echo".to_string(),
            namespace: Some("tools".to_string()),
        }),
        description: "echo input".to_string(),
        kind: proto::ToolKind::Function as i32,
        input_schema_json: Some(br#"{"type":"object"}"#.to_vec()),
        output_schema_json: None,
    };
    let definition = tool_definition(base.clone()).expect("valid function definition");
    assert_eq!(definition.identity.name, "echo");
    assert_eq!(definition.identity.namespace.as_deref(), Some("tools"));

    let missing_name = tool_definition(proto::ToolDefinition {
        tool_name: None,
        ..base.clone()
    });
    assert!(missing_name.is_err());

    let freeform_kind = tool_definition(proto::ToolDefinition {
        kind: proto::ToolKind::Freeform as i32,
        ..base.clone()
    });
    assert!(freeform_kind.is_ok());

    let unsupported_kind = tool_definition(proto::ToolDefinition {
        kind: proto::ToolKind::Unspecified as i32,
        ..base.clone()
    });
    assert!(unsupported_kind.is_err());

    let invalid_schema = tool_definition(proto::ToolDefinition {
        input_schema_json: Some(b"not-json".to_vec()),
        ..base
    });
    assert!(invalid_schema.is_err());
}
