//! Session event and delegate callback routing.
//!
//! The concrete callback methods remain on `GrpcCodeModeSession` so they can
//! share the session's callback maps. This module owns callback policy
//! constants and keeps the routing boundary explicit for future transports.

use super::GrpcCodeModeSession;
use code_mode_protocol::grpc::{self as proto, code_mode_host_client::CodeModeHostClient};
use code_mode_protocol::{CodeModeToolKind, RuntimeCodeModeCellId, RuntimeCodeModeNestedToolCall};
use tokio_util::sync::CancellationToken;

pub(super) async fn tool_call(
    session: &GrpcCodeModeSession,
    call: proto::ToolCall,
) -> Result<(), String> {
    if call.session_id != session.session_id {
        return Err(format!(
            "code-mode gRPC tool call belongs to session {} instead of {}",
            call.session_id, session.session_id
        ));
    }
    super::validate_identifier(&call.invocation_id, "tool invocation ID")?;
    super::validate_identifier(&call.cell_id, "cell ID")?;
    super::validate_identifier(&call.execution_id, "execution ID")?;
    super::validate_identifier(&call.runtime_tool_call_id, "runtime tool call ID")?;
    if session
        .cancelled_callbacks
        .lock()
        .await
        .remove(&call.invocation_id)
    {
        return Ok(());
    }
    let tool_name = call
        .tool_name
        .ok_or_else(|| "code-mode gRPC tool call omitted name".to_string())?;
    super::validate_identifier(&tool_name.name, "tool name")?;
    if let Some(namespace) = tool_name.namespace.as_deref() {
        super::validate_identifier(namespace, "tool namespace")?;
    }
    let input = call
        .input_json
        .map(|value| serde_json::from_slice(&value))
        .transpose()
        .map_err(|error| format!("invalid code-mode gRPC tool input: {error}"))?;
    let invocation = RuntimeCodeModeNestedToolCall {
        cell_id: RuntimeCodeModeCellId::new(call.cell_id),
        runtime_tool_call_id: call.runtime_tool_call_id,
        tool_name: tool_name.name,
        kind: match proto::ToolKind::try_from(call.tool_kind) {
            Ok(proto::ToolKind::Function) => CodeModeToolKind::Function,
            Ok(proto::ToolKind::Freeform) => CodeModeToolKind::Freeform,
            Ok(proto::ToolKind::Unspecified) | Err(_) => {
                return Err("invalid code-mode gRPC tool kind".to_string())
            }
        },
        input,
    };
    let cancellation = CancellationToken::new();
    session
        .pending_callbacks
        .lock()
        .await
        .insert(call.invocation_id.clone(), cancellation.clone());
    let result = session
        .delegate
        .invoke_tool(invocation, cancellation.clone())
        .await;
    session
        .pending_callbacks
        .lock()
        .await
        .remove(&call.invocation_id);
    if cancellation.is_cancelled() {
        return Ok(());
    }
    let outcome = match result {
        Ok(value) => {
            proto::complete_tool_call_request::Outcome::Succeeded(proto::ToolCallSucceeded {
                output_json: serde_json::to_vec(&value).map_err(|error| error.to_string())?,
            })
        }
        Err(error) => proto::complete_tool_call_request::Outcome::Failed(proto::ToolCallFailed {
            message: error,
        }),
    };
    CodeModeHostClient::new(session.channel.clone())
        .complete_tool_call(proto::CompleteToolCallRequest {
            session_id: session.session_id.clone(),
            invocation_id: call.invocation_id,
            outcome: Some(outcome),
        })
        .await
        .map_err(|error| format!("failed to complete code-mode gRPC tool call: {error}"))?;
    Ok(())
}

pub(super) async fn session_event(
    session: &GrpcCodeModeSession,
    event: proto::SessionEvent,
) -> Result<(), String> {
    match event.event {
        Some(proto::session_event::Event::ToolCallCancelled(cancelled)) => {
            cancel_pending(session, &cancelled.invocation_id).await;
        }
        Some(proto::session_event::Event::Notification(notification)) => {
            handle_notification(session, notification).await?;
        }
        Some(proto::session_event::Event::NotificationCancelled(cancelled)) => {
            cancel_pending(session, &cancelled.notification_id).await;
        }
        Some(proto::session_event::Event::CellClosed(closed)) => {
            session
                .delegate
                .cell_closed(&RuntimeCodeModeCellId::new(closed.cell_id));
        }
        Some(proto::session_event::Event::Opened(_)) | None => {}
    }
    Ok(())
}

async fn cancel_pending(session: &GrpcCodeModeSession, id: &str) {
    if let Some(token) = session.pending_callbacks.lock().await.remove(id) {
        token.cancel();
    } else {
        session
            .cancelled_callbacks
            .lock()
            .await
            .insert(id.to_string());
    }
}

pub(super) async fn handle_notification(
    session: &GrpcCodeModeSession,
    notification: proto::Notification,
) -> Result<(), String> {
    super::validate_identifier(&notification.notification_id, "notification ID")?;
    super::validate_identifier(&notification.execution_id, "execution ID")?;
    super::validate_identifier(&notification.cell_id, "cell ID")?;
    super::validate_identifier(&notification.call_id, "notification call ID")?;
    if session
        .cancelled_callbacks
        .lock()
        .await
        .remove(&notification.notification_id)
    {
        return Ok(());
    }
    let cancellation = CancellationToken::new();
    session
        .pending_callbacks
        .lock()
        .await
        .insert(notification.notification_id.clone(), cancellation.clone());
    let result = session
        .delegate
        .notify(
            notification.call_id,
            RuntimeCodeModeCellId::new(notification.cell_id),
            notification.text,
            cancellation.clone(),
        )
        .await;
    session
        .pending_callbacks
        .lock()
        .await
        .remove(&notification.notification_id);
    if cancellation.is_cancelled() {
        return Ok(());
    }
    result?;
    CodeModeHostClient::new(session.channel.clone())
        .acknowledge_notification(proto::AcknowledgeNotificationRequest {
            session_id: session.session_id.clone(),
            notification_id: notification.notification_id,
        })
        .await
        .map_err(|error| format!("failed to acknowledge code-mode notification: {error}"))?;
    Ok(())
}
