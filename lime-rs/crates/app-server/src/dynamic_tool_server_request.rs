use crate::approval_server_request::WaitServerRequestError;
use crate::{AppServer, DynamicToolRespondRequest};
use app_server_protocol::protocol::v2::{
    DynamicToolCallOutputContentItem, DynamicToolCallParams, DynamicToolCallResponse,
    METHOD_ITEM_TOOL_CALL,
};
use app_server_protocol::AgentEvent;
use serde_json::Value;

impl AppServer {
    pub(crate) async fn handle_dynamic_tool_server_request(&self, event: AgentEvent) -> bool {
        let request = match dynamic_tool_server_request(&event) {
            Ok(Some(request)) => request,
            Ok(None) => return false,
            Err(error) => {
                tracing::warn!(event_id = %event.event_id, %error, "invalid dynamic tool server request event");
                return true;
            }
        };
        let response = match self
            .wait_server_request::<_, DynamicToolCallResponse>(
                METHOD_ITEM_TOOL_CALL,
                &request.params.thread_id,
                request.params.clone(),
            )
            .await
        {
            Ok(response) => validate_response(response),
            Err(WaitServerRequestError::Transition) => return true,
            Err(WaitServerRequestError::Failed(error)) => {
                tracing::warn!(%error, "dynamic tool server request failed closed");
                failed_response(format!("dynamic tool host response failed: {error}"))
            }
        };
        if let Err(error) = self
            .processor
            .runtime()
            .respond_dynamic_tool(DynamicToolRespondRequest {
                session_id: request.session_id,
                thread_id: request.params.thread_id,
                turn_id: request.params.turn_id,
                call_id: request.params.call_id,
                response,
            })
            .await
        {
            tracing::warn!(event_id = %event.event_id, %error, "typed dynamic tool response rejected by runtime");
        }
        true
    }
}

struct DynamicToolServerRequest {
    session_id: String,
    params: DynamicToolCallParams,
}

fn dynamic_tool_server_request(
    event: &AgentEvent,
) -> Result<Option<DynamicToolServerRequest>, String> {
    if event.event_type != "dynamic_tool.requested" {
        return Ok(None);
    }
    let session_id = required_text(&event.session_id, "sessionId")?;
    let thread_id = required_optional_text(event.thread_id.as_deref(), "threadId")?;
    let turn_id = required_optional_text(event.turn_id.as_deref(), "turnId")?;
    let call_id = required_payload_text(&event.payload, "call_id", "callId")?;
    let tool = required_payload_text(&event.payload, "tool", "tool")?;
    let namespace = match event.payload.get("namespace") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) => Some(required_text(value, "namespace")?),
        Some(_) => return Err("dynamic_tool.requested namespace is invalid".to_string()),
    };
    let arguments = event
        .payload
        .get("arguments")
        .cloned()
        .ok_or_else(|| "dynamic_tool.requested has no arguments".to_string())?;
    if !arguments.is_object() {
        return Err("dynamic_tool.requested arguments must be an object".to_string());
    }
    Ok(Some(DynamicToolServerRequest {
        session_id,
        params: DynamicToolCallParams {
            thread_id,
            turn_id,
            call_id,
            namespace,
            tool,
            arguments,
        },
    }))
}

fn validate_response(response: DynamicToolCallResponse) -> DynamicToolCallResponse {
    let invalid = response.content_items.iter().find_map(|item| match item {
        DynamicToolCallOutputContentItem::InputText { .. } => None,
        DynamicToolCallOutputContentItem::InputImage { image_url }
            if !image_url.starts_with("data:image/") =>
        {
            Some("dynamic tool image output must use an inline data:image URL")
        }
        DynamicToolCallOutputContentItem::InputAudio { audio_url }
            if !audio_url.starts_with("data:audio/") =>
        {
            Some("dynamic tool audio output must use an inline data:audio URL")
        }
        _ => None,
    });
    invalid.map(failed_response).unwrap_or(response)
}

fn failed_response(message: impl Into<String>) -> DynamicToolCallResponse {
    DynamicToolCallResponse {
        content_items: vec![DynamicToolCallOutputContentItem::InputText {
            text: message.into(),
        }],
        success: false,
    }
}

fn required_payload_text(payload: &Value, key: &str, field: &str) -> Result<String, String> {
    payload
        .get(key)
        .and_then(Value::as_str)
        .map(|value| required_text(value, field))
        .transpose()?
        .ok_or_else(|| format!("dynamic_tool.requested has no {field}"))
}

fn required_text(value: &str, field: &str) -> Result<String, String> {
    let value = value.trim();
    (!value.is_empty())
        .then(|| value.to_string())
        .ok_or_else(|| format!("dynamic_tool.requested has no {field}"))
}

fn required_optional_text(value: Option<&str>, field: &str) -> Result<String, String> {
    value
        .map(|value| required_text(value, field))
        .transpose()?
        .ok_or_else(|| format!("dynamic_tool.requested has no {field}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_exact_identity_and_fails_remote_media_closed() {
        let event = AgentEvent {
            event_id: "event-1".to_string(),
            session_id: "session-1".to_string(),
            thread_id: Some("thread-1".to_string()),
            turn_id: Some("turn-1".to_string()),
            sequence: 1,
            event_type: "dynamic_tool.requested".to_string(),
            timestamp: "2026-07-30T00:00:00Z".to_string(),
            payload: serde_json::json!({
                "call_id": "call-1",
                "namespace": "desktop",
                "tool": "appInfo",
                "arguments": {}
            }),
        };
        let request = dynamic_tool_server_request(&event)
            .expect("parse")
            .expect("request");
        assert_eq!(request.params.call_id, "call-1");
        assert_eq!(request.params.namespace.as_deref(), Some("desktop"));

        let response = validate_response(DynamicToolCallResponse {
            content_items: vec![DynamicToolCallOutputContentItem::InputImage {
                image_url: "https://example.com/image.png".to_string(),
            }],
            success: true,
        });
        assert!(!response.success);
        assert!(matches!(
            &response.content_items[0],
            DynamicToolCallOutputContentItem::InputText { text }
                if text.contains("data:image")
        ));
    }
}
