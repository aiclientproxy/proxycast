//! Runtime delegate implementation shared by host transports.
//!
//! A gRPC session owns the callback lease, while this delegate only translates
//! runtime callbacks into routed session events.  The weak reference prevents
//! a runtime callback from keeping a closed gRPC session alive.

use crate::grpc::{events, routing, session::GrpcSession};
use code_mode_protocol::grpc as proto;
use code_mode_protocol::{
    RuntimeCodeModeCellId, RuntimeCodeModeFuture, RuntimeCodeModeNestedToolCall,
    RuntimeCodeModeSessionDelegate,
};
use serde_json::Value;
use std::sync::Weak;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

pub(crate) struct GrpcDelegate {
    pub(crate) session: Weak<GrpcSession>,
}

impl RuntimeCodeModeSessionDelegate for GrpcDelegate {
    fn invoke_tool<'a>(
        &'a self,
        invocation: RuntimeCodeModeNestedToolCall,
        cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, Value> {
        Box::pin(async move {
            let session = self
                .session
                .upgrade()
                .ok_or_else(|| "code-mode session is closed".to_string())?;
            tokio::select! {
                result = routing::publish_tool_call(&session, invocation, cancellation_token.clone()) => result,
                _ = cancellation_token.cancelled() => Err("code-mode tool invocation cancelled".to_string()),
            }
        })
    }

    fn notify<'a>(
        &'a self,
        tool_call_id: String,
        cell_id: RuntimeCodeModeCellId,
        text: String,
        cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, ()> {
        Box::pin(async move {
            let session = self
                .session
                .upgrade()
                .ok_or_else(|| "code-mode session is closed".to_string())?;
            if cancellation_token.is_cancelled() {
                return Err("code-mode notification cancelled".to_string());
            }
            let notification_id = uuid::Uuid::new_v4().to_string();
            let execution_id = session
                .execution_id_for_cell(cell_id.as_str())
                .await
                .ok_or_else(|| "code-mode cell has no owning execution".to_string())?;
            let (sender, receiver) = oneshot::channel();
            let mut pending = session.pending_notifications.lock().await;
            if pending.len() >= code_mode_protocol::host::MAX_PENDING_DELEGATE_CALLS {
                return Err("code-mode notification request limit exceeded".to_string());
            }
            pending.insert(notification_id.clone(), sender);
            drop(pending);
            if let Err(error) = session
                .publish_event(proto::session_event::Event::Notification(
                    proto::Notification {
                        notification_id: notification_id.clone(),
                        execution_id,
                        cell_id: cell_id.to_string(),
                        call_id: tool_call_id,
                        text,
                    },
                ))
                .await
            {
                session
                    .pending_notifications
                    .lock()
                    .await
                    .remove(&notification_id);
                return Err(error);
            }
            tokio::select! {
                result = receiver => result.map_err(|_| "code-mode notification completion channel closed".to_string())?,
                _ = cancellation_token.cancelled() => {
                    if session.pending_notifications.lock().await.remove(&notification_id).is_some() {
                        events::notification_cancelled(&session, notification_id).await;
                    }
                    Err("code-mode notification cancelled".to_string())
                }
            }
        })
    }

    fn cell_closed(&self, cell_id: &RuntimeCodeModeCellId) {
        let Some(session) = self.session.upgrade() else {
            return;
        };
        let cell_id = cell_id.clone();
        tokio::spawn(async move {
            events::cell_closed(&session, cell_id).await;
        });
    }
}
