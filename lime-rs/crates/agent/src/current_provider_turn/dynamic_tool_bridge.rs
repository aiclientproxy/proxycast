use super::mcp_step_snapshot::DynamicToolRoute;
use crate::protocol::AgentEvent;
use agent_protocol::ThreadId;
use agent_runtime::session_loop::{RuntimeSessionInputHandle, RuntimeSessionResponseKind};
use app_server_protocol::protocol::v2::{
    DynamicToolCallOutputContentItem, DynamicToolCallResponse,
};
use std::collections::HashMap;
use tokio::sync::mpsc::UnboundedSender;
use tool_runtime::tool_executor::{
    RuntimeToolExecutionError, RuntimeToolExecutionRequest, RuntimeToolExecutionResult,
    RuntimeToolPolicyErrorKind,
};

const CONTENT_ITEMS_METADATA_KEY: &str = "dynamic_tool_content_items";
const ROUTE_METADATA_KEY: &str = "dynamic_tool";

pub(super) async fn call_dynamic_tool(
    request: RuntimeToolExecutionRequest<'_>,
    thread_id: &ThreadId,
    route: DynamicToolRoute,
    pending_input: Option<RuntimeSessionInputHandle>,
    event_sender: &UnboundedSender<AgentEvent>,
) -> Result<RuntimeToolExecutionResult, RuntimeToolExecutionError> {
    let identity = request.context.tool_identity().ok_or_else(|| {
        dynamic_tool_error(
            "dynamic tool requires canonical tool identity",
            "dynamic_tool_identity_missing",
        )
        .before_handler()
    })?;
    if identity.turn_id().trim().is_empty()
        || identity.call_id().trim().is_empty()
        || request.context.session_id().trim().is_empty()
        || thread_id.as_str().trim().is_empty()
    {
        return Err(dynamic_tool_error(
            "dynamic tool requires canonical session/thread/turn/call identity",
            "dynamic_tool_identity_missing",
        )
        .before_handler());
    }
    let response_handle = pending_input.ok_or_else(|| {
        dynamic_tool_error(
            "dynamic tool requires the active session response owner",
            "session_response_owner_missing",
        )
        .before_handler()
    })?;
    let pending = response_handle
        .register_response(RuntimeSessionResponseKind::DynamicTool, identity.call_id())
        .await
        .map_err(|error| {
            dynamic_tool_error(error.message, "dynamic_tool_response_registration").before_handler()
        })?;
    event_sender
        .send(AgentEvent::DynamicToolCallRequested {
            call_id: identity.call_id().to_string(),
            namespace: route.namespace.clone(),
            tool: route.tool.clone(),
            arguments: request.params.clone(),
        })
        .map_err(|_| {
            dynamic_tool_error(
                "dynamic tool request event channel is closed",
                "dynamic_tool_event_channel_closed",
            )
            .before_handler()
        })?;
    let response = pending
        .wait()
        .await
        .map_err(|error| dynamic_tool_error(error.message, "dynamic_tool_response_wait_failed"))?;
    let response =
        serde_json::from_value::<DynamicToolCallResponse>(response).map_err(|error| {
            dynamic_tool_error(
                format!("dynamic tool response is invalid: {error}"),
                "dynamic_tool_response_invalid",
            )
        })?;
    validate_content_items(&response.content_items)?;

    let output = serde_json::to_string(&response.content_items).map_err(|error| {
        dynamic_tool_error(
            format!("failed to serialize dynamic tool response: {error}"),
            "dynamic_tool_response_invalid",
        )
    })?;
    let error = (!response.success).then(|| dynamic_tool_failure_text(&response.content_items));
    let structured_content = serde_json::to_value(&response.content_items).map_err(|error| {
        dynamic_tool_error(
            format!("failed to project dynamic tool response: {error}"),
            "dynamic_tool_response_invalid",
        )
    })?;
    let metadata = HashMap::from([
        (
            ROUTE_METADATA_KEY.to_string(),
            serde_json::json!({
                "runtimeToolName": route.runtime_tool_name,
                "namespace": route.namespace,
                "tool": route.tool,
            }),
        ),
        (
            CONTENT_ITEMS_METADATA_KEY.to_string(),
            structured_content.clone(),
        ),
    ]);
    Ok(
        RuntimeToolExecutionResult::new(response.success, output, error, metadata)
            .with_structured_content(structured_content),
    )
}

fn validate_content_items(
    content_items: &[DynamicToolCallOutputContentItem],
) -> Result<(), RuntimeToolExecutionError> {
    for item in content_items {
        match item {
            DynamicToolCallOutputContentItem::InputText { .. } => {}
            DynamicToolCallOutputContentItem::InputImage { image_url } => {
                if !image_url.starts_with("data:") {
                    return Err(dynamic_tool_error(
                        "dynamic tool image output must use a data URL",
                        "dynamic_tool_remote_image_rejected",
                    ));
                }
            }
            DynamicToolCallOutputContentItem::InputAudio { audio_url } => {
                if !audio_url.starts_with("data:") {
                    return Err(dynamic_tool_error(
                        "dynamic tool audio output must use a data URL",
                        "dynamic_tool_remote_audio_rejected",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn dynamic_tool_failure_text(content_items: &[DynamicToolCallOutputContentItem]) -> String {
    content_items
        .iter()
        .filter_map(|item| match item {
            DynamicToolCallOutputContentItem::InputText { text } => Some(text.trim()),
            _ => None,
        })
        .find(|text| !text.is_empty())
        .unwrap_or("dynamic tool call failed")
        .to_string()
}

fn dynamic_tool_error(message: impl Into<String>, code: &str) -> RuntimeToolExecutionError {
    RuntimeToolExecutionError::new(
        message,
        Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
            code.to_string(),
        )),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_media_is_rejected() {
        assert!(
            validate_content_items(&[DynamicToolCallOutputContentItem::InputImage {
                image_url: "https://example.com/image.png".to_string(),
            }])
            .is_err()
        );
        assert!(
            validate_content_items(&[DynamicToolCallOutputContentItem::InputAudio {
                audio_url: "https://example.com/audio.wav".to_string(),
            }])
            .is_err()
        );
        assert!(
            validate_content_items(&[DynamicToolCallOutputContentItem::InputImage {
                image_url: "data:image/png;base64,AA==".to_string(),
            }])
            .is_ok()
        );
    }
}
