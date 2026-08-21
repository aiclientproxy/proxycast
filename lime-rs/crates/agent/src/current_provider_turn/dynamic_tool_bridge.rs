use super::{mcp_step_snapshot::DynamicToolRoute, tool_executor::orchestration};
use crate::protocol::AgentEvent;
use crate::runtime_state::AgentRuntimeState;
use agent_protocol::ThreadId;
use agent_runtime::session_loop::{RuntimeSessionInputHandle, RuntimeSessionResponseKind};
use app_server_protocol::protocol::v2::{
    DynamicToolCallApproval, DynamicToolCallOutputContentItem, DynamicToolCallPhase,
    DynamicToolCallResponse,
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
    state: &AgentRuntimeState,
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
    let mut response = request_dynamic_tool_phase(
        request,
        &route,
        &response_handle,
        event_sender,
        DynamicToolCallPhase::Preflight,
        None,
    )
    .await?;
    if let Some(approval) = response.approval.take() {
        if route.namespace.as_deref() != Some("browser") {
            return Err(dynamic_tool_error(
                "dynamic tool approval is only supported for Browser actions",
                "dynamic_tool_approval_namespace_invalid",
            ));
        }
        validate_approval(&approval)?;
        orchestration::wait_for_browser_action_approval(
            state,
            event_sender,
            request,
            thread_id,
            Some(&response_handle),
            &approval,
        )
        .await?;
        response = request_dynamic_tool_phase(
            request,
            &route,
            &response_handle,
            event_sender,
            DynamicToolCallPhase::ApprovedExecute,
            Some(approval.approval_token),
        )
        .await?;
        if response.approval.is_some() {
            return Err(dynamic_tool_error(
                "approved dynamic tool execution cannot request another approval",
                "dynamic_tool_approval_repeated",
            ));
        }
    }
    validate_content_items(&response.content_items)?;
    project_dynamic_tool_response(route, response)
}

async fn request_dynamic_tool_phase(
    request: RuntimeToolExecutionRequest<'_>,
    route: &DynamicToolRoute,
    response_handle: &RuntimeSessionInputHandle,
    event_sender: &UnboundedSender<AgentEvent>,
    phase: DynamicToolCallPhase,
    approval_token: Option<String>,
) -> Result<DynamicToolCallResponse, RuntimeToolExecutionError> {
    let identity = request
        .context
        .tool_identity()
        .expect("dynamic tool identity is validated before dispatch");
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
            phase,
            approval_token,
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
    serde_json::from_value::<DynamicToolCallResponse>(response).map_err(|error| {
        dynamic_tool_error(
            format!("dynamic tool response is invalid: {error}"),
            "dynamic_tool_response_invalid",
        )
    })
}

fn project_dynamic_tool_response(
    route: DynamicToolRoute,
    response: DynamicToolCallResponse,
) -> Result<RuntimeToolExecutionResult, RuntimeToolExecutionError> {
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

fn validate_approval(approval: &DynamicToolCallApproval) -> Result<(), RuntimeToolExecutionError> {
    let required = [
        approval.approval_token.as_str(),
        approval.reason.as_str(),
        approval.risk_class.as_str(),
        approval.action_kind.as_str(),
        approval.browser_session_id.as_str(),
        approval.tab_id.as_str(),
        approval.view_id.as_str(),
        approval.snapshot_id.as_str(),
    ];
    if required.iter().any(|value| value.trim().is_empty()) || approval.web_contents_id == 0 {
        return Err(dynamic_tool_error(
            "dynamic tool approval descriptor is incomplete",
            "dynamic_tool_approval_invalid",
        ));
    }
    Ok(())
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
