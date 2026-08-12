use crate::code_mode::v8::protocol::ExecuteRequest;
use crate::code_mode::v8::protocol::FunctionCallOutputContentItem;
use crate::code_mode::v8::protocol::ImageDetail;
use crate::code_mode::v8::protocol::ToolDefinition;
use crate::code_mode::v8::protocol::ToolName;

use crate::code_mode::v8::session_runtime::CreateCellRequest as CellRequest;
use crate::code_mode::v8::session_runtime::ImageDetail as CellImageDetail;
use crate::code_mode::v8::session_runtime::OutputItem as CellOutputItem;
pub(super) fn runtime_request(request: CellRequest) -> ExecuteRequest {
    ExecuteRequest {
        tool_call_id: request.tool_call_id,
        enabled_tools: request
            .enabled_tools
            .into_iter()
            .map(|definition| ToolDefinition {
                name: definition.name,
                tool_name: ToolName {
                    name: definition.tool_name.name,
                    namespace: definition.tool_name.namespace,
                },
                description: definition.description,
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
