use super::conversion::response_from_outcome;
use code_mode_protocol::grpc as proto;
use code_mode_protocol::{
    FunctionCallOutputContentItem, ImageDetail, RuntimeCodeModeCellId, RuntimeCodeModeResponse,
};
use std::time::Duration;

#[test]
fn decodes_text_image_audio_and_host_duration() {
    let outcome = proto::ExecutionOutcome {
        cell_id: "cell".to_string(),
        content_items: vec![
            proto::ContentItem {
                item: Some(proto::content_item::Item::Text(proto::TextContent {
                    text: "hello".to_string(),
                })),
            },
            proto::ContentItem {
                item: Some(proto::content_item::Item::Image(proto::ImageContent {
                    image_url: "data:image/png;base64,AA==".to_string(),
                    detail: Some(proto::ImageDetail::Original as i32),
                })),
            },
            proto::ContentItem {
                item: Some(proto::content_item::Item::Audio(proto::AudioContent {
                    audio_url: "data:audio/wav;base64,AA==".to_string(),
                })),
            },
        ],
        code_mode_host_duration_ns: 42,
        outcome: Some(proto::execution_outcome::Outcome::Completed(
            proto::ExecutionCompleted {
                error_text: Some("warning".to_string()),
            },
        )),
    };
    assert_eq!(
        response_from_outcome(outcome).expect("typed response"),
        RuntimeCodeModeResponse::Result {
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
            error_text: Some("warning".to_string()),
            code_mode_host_duration: Some(Duration::from_nanos(42)),
        }
    );
}

#[test]
fn rejects_missing_content_oneof_and_invalid_image_detail() {
    let missing_content = proto::ExecutionOutcome {
        cell_id: "cell".to_string(),
        content_items: vec![proto::ContentItem { item: None }],
        code_mode_host_duration_ns: 0,
        outcome: Some(proto::execution_outcome::Outcome::Yielded(
            proto::ExecutionYielded {},
        )),
    };
    assert!(response_from_outcome(missing_content).is_err());

    let invalid_detail = proto::ExecutionOutcome {
        cell_id: "cell".to_string(),
        content_items: vec![proto::ContentItem {
            item: Some(proto::content_item::Item::Image(proto::ImageContent {
                image_url: "image".to_string(),
                detail: Some(proto::ImageDetail::Unspecified as i32),
            })),
        }],
        code_mode_host_duration_ns: 0,
        outcome: Some(proto::execution_outcome::Outcome::Yielded(
            proto::ExecutionYielded {},
        )),
    };
    assert!(response_from_outcome(invalid_detail).is_err());
}
