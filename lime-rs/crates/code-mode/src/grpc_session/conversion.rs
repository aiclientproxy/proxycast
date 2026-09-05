//! Conversion between protobuf execution events and the protocol response.

use code_mode_protocol::grpc as proto;
use code_mode_protocol::{
    FunctionCallOutputContentItem, ImageDetail, RuntimeCodeModeCellId, RuntimeCodeModeResponse,
};
use std::time::Duration;

pub(super) fn response_from_outcome(
    outcome: proto::ExecutionOutcome,
) -> Result<RuntimeCodeModeResponse, String> {
    super::validate_identifier(&outcome.cell_id, "cell ID")?;
    let cell_id = RuntimeCodeModeCellId::new(outcome.cell_id);
    let content_items = outcome
        .content_items
        .into_iter()
        .map(content_item)
        .collect::<Result<Vec<_>, _>>()?;
    let code_mode_host_duration = Some(Duration::from_nanos(outcome.code_mode_host_duration_ns));
    match outcome
        .outcome
        .ok_or_else(|| "code-mode execution outcome omitted state".to_string())?
    {
        proto::execution_outcome::Outcome::Yielded(_) => Ok(RuntimeCodeModeResponse::Yielded {
            cell_id,
            content_items,
            code_mode_host_duration,
        }),
        proto::execution_outcome::Outcome::Terminated(_) => {
            Ok(RuntimeCodeModeResponse::Terminated {
                cell_id,
                content_items,
                code_mode_host_duration,
            })
        }
        proto::execution_outcome::Outcome::Completed(completed) => {
            Ok(RuntimeCodeModeResponse::Result {
                cell_id,
                content_items,
                error_text: completed.error_text,
                code_mode_host_duration,
            })
        }
    }
}

fn content_item(item: proto::ContentItem) -> Result<FunctionCallOutputContentItem, String> {
    match item
        .item
        .ok_or_else(|| "code-mode content item omitted value".to_string())?
    {
        proto::content_item::Item::Text(text) => {
            Ok(FunctionCallOutputContentItem::InputText { text: text.text })
        }
        proto::content_item::Item::Image(image) => Ok(FunctionCallOutputContentItem::InputImage {
            image_url: image.image_url,
            detail: image.detail.map(image_detail).transpose()?,
        }),
        proto::content_item::Item::Audio(audio) => Ok(FunctionCallOutputContentItem::InputAudio {
            audio_url: audio.audio_url,
        }),
    }
}

fn image_detail(detail: i32) -> Result<ImageDetail, String> {
    match proto::ImageDetail::try_from(detail) {
        Ok(proto::ImageDetail::Auto) => Ok(ImageDetail::Auto),
        Ok(proto::ImageDetail::Low) => Ok(ImageDetail::Low),
        Ok(proto::ImageDetail::High) => Ok(ImageDetail::High),
        Ok(proto::ImageDetail::Original) => Ok(ImageDetail::Original),
        Ok(proto::ImageDetail::Unspecified) | Err(_) => {
            Err("code-mode image content has invalid detail".to_string())
        }
    }
}
