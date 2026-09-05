//! Conversion between host runtime values and the protobuf wire contract.

use code_mode_protocol::grpc as proto;
use code_mode_protocol::{
    CodeModeToolKind, FunctionCallOutputContentItem, ImageDetail, RuntimeCodeModeResponse,
    RuntimeCodeModeTool, RuntimeToolDefinition, RuntimeToolIdentity,
};
use std::time::Duration;
use tonic::Status;

use super::validation::{require_identifier, tool_name};

pub(super) fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

pub(super) fn tool_definition(tool: proto::ToolDefinition) -> Result<RuntimeCodeModeTool, Status> {
    require_identifier(&tool.name, "tool name")?;
    let identity = tool
        .tool_name
        .ok_or_else(|| Status::invalid_argument("tool definition omitted its tool name"))?;
    tool_name(&identity)?;
    let kind = match proto::ToolKind::try_from(tool.kind) {
        Ok(proto::ToolKind::Function) => CodeModeToolKind::Function,
        Ok(proto::ToolKind::Freeform) => CodeModeToolKind::Freeform,
        Ok(proto::ToolKind::Unspecified) | Err(_) => {
            return Err(Status::invalid_argument("invalid tool kind"));
        }
    };
    let name = identity.name;
    let namespace = identity.namespace;
    if let Some(output_schema_json) = tool.output_schema_json.as_deref() {
        serde_json::from_slice::<serde_json::Value>(output_schema_json).map_err(|error| {
            Status::invalid_argument(format!("invalid tool output schema JSON: {error}"))
        })?;
    }
    let input_schema = tool
        .input_schema_json
        .map(|json| {
            serde_json::from_slice(&json).map_err(|error| {
                Status::invalid_argument(format!("invalid tool input schema JSON: {error}"))
            })
        })
        .transpose()?
        .unwrap_or_else(|| serde_json::json!({}));
    Ok(RuntimeCodeModeTool {
        identity: RuntimeToolIdentity { namespace, name },
        definition: RuntimeToolDefinition::new(tool.name.clone(), tool.description, input_schema),
        kind,
        code_name: tool.name.clone(),
        global_name: tool.name,
    })
}

pub(super) fn execution_outcome(
    response: RuntimeCodeModeResponse,
    duration_ns: u64,
) -> proto::ExecutionOutcome {
    let response = response.with_code_mode_host_duration(Duration::from_nanos(duration_ns));
    let (cell_id, content_items, outcome) = match response {
        RuntimeCodeModeResponse::Yielded {
            cell_id,
            content_items,
            ..
        } => (
            cell_id,
            content_items,
            proto::execution_outcome::Outcome::Yielded(proto::ExecutionYielded {}),
        ),
        RuntimeCodeModeResponse::Terminated {
            cell_id,
            content_items,
            ..
        } => (
            cell_id,
            content_items,
            proto::execution_outcome::Outcome::Terminated(proto::ExecutionTerminated {}),
        ),
        RuntimeCodeModeResponse::Result {
            cell_id,
            content_items,
            error_text,
            ..
        } => (
            cell_id,
            content_items,
            proto::execution_outcome::Outcome::Completed(proto::ExecutionCompleted { error_text }),
        ),
    };
    proto::ExecutionOutcome {
        cell_id: cell_id.to_string(),
        content_items: content_items.into_iter().map(content_item).collect(),
        code_mode_host_duration_ns: duration_ns,
        outcome: Some(outcome),
    }
}

fn content_item(item: FunctionCallOutputContentItem) -> proto::ContentItem {
    let item = match item {
        FunctionCallOutputContentItem::InputText { text } => {
            proto::content_item::Item::Text(proto::TextContent { text })
        }
        FunctionCallOutputContentItem::InputImage { image_url, detail } => {
            proto::content_item::Item::Image(proto::ImageContent {
                image_url,
                detail: detail.map(image_detail),
            })
        }
        FunctionCallOutputContentItem::InputAudio { audio_url } => {
            proto::content_item::Item::Audio(proto::AudioContent { audio_url })
        }
    };
    proto::ContentItem { item: Some(item) }
}

fn image_detail(detail: ImageDetail) -> i32 {
    match detail {
        ImageDetail::Auto => proto::ImageDetail::Auto as i32,
        ImageDetail::Low => proto::ImageDetail::Low as i32,
        ImageDetail::High => proto::ImageDetail::High as i32,
        ImageDetail::Original => proto::ImageDetail::Original as i32,
    }
}
