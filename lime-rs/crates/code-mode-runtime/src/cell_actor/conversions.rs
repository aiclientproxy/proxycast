use crate::runtime::{RuntimeRequest, RuntimeToolMetadata};
use crate::session_runtime::CreateCellRequest as CellRequest;
use crate::session_runtime::ImageDetail as CellImageDetail;
use crate::session_runtime::OutputItem as CellOutputItem;
use code_mode_protocol::{FunctionCallOutputContentItem, ImageDetail};

pub(super) fn runtime_request(request: CellRequest) -> RuntimeRequest {
    RuntimeRequest {
        tool_call_id: request.tool_call_id,
        enabled_tools: request
            .enabled_tools
            .into_iter()
            .map(|definition| RuntimeToolMetadata {
                global_name: definition.name,
                tool_name: definition.tool_name,
                description: definition.description,
                kind: definition.kind,
            })
            .collect(),
        source: request.source,
    }
}

pub(super) fn output_item(item: FunctionCallOutputContentItem) -> CellOutputItem {
    match item {
        FunctionCallOutputContentItem::InputText { text } => CellOutputItem::Text { text },
        FunctionCallOutputContentItem::InputImage { image_url, detail } => CellOutputItem::Image {
            image_url,
            detail: detail.map(|detail| match detail {
                ImageDetail::Auto => CellImageDetail::Auto,
                ImageDetail::Low => CellImageDetail::Low,
                ImageDetail::High => CellImageDetail::High,
                ImageDetail::Original => CellImageDetail::Original,
            }),
        },
        FunctionCallOutputContentItem::InputAudio { audio_url } => {
            CellOutputItem::Audio { audio_url }
        }
    }
}
