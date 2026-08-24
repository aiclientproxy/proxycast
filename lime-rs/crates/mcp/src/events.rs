//! MCP 事件 Payload。

use crate::types::McpToolDefinition;
use serde::Serialize;
use serde_json::Value;
use tokio::sync::broadcast;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

/// 工具列表更新事件
#[derive(Debug, Clone, Serialize)]
pub struct McpToolsUpdatedPayload {
    pub tools: Vec<McpToolDefinition>,
}

/// 资源列表更新事件
#[derive(Debug, Clone, Serialize)]
pub struct McpResourcesUpdatedPayload {
    pub server_name: String,
}

/// 资源内容更新事件
#[derive(Debug, Clone, Serialize)]
pub struct McpResourceUpdatedPayload {
    pub server_name: String,
    pub uri: String,
}

/// Server-originated notification received by a session-owned MCP client.
///
/// The raw method and params are preserved so App Server can apply the
/// Codex event-stream envelope without making assumptions about a hosted app's
/// notification schema.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct McpServerNotification {
    pub server_name: String,
    pub method: String,
    pub params: Value,
}

/// A session-owned MCP `events/stream` request and its server notifications.
///
/// Dropping the value cancels the pending remote request, which keeps runtime
/// generation replacement from leaving an event-stream request behind.
pub struct McpEventStream {
    receiver: broadcast::Receiver<McpServerNotification>,
    request_done: oneshot::Receiver<Result<(), String>>,
    cancellation: CancellationToken,
}

impl McpEventStream {
    pub(crate) fn new(
        receiver: broadcast::Receiver<McpServerNotification>,
        request_done: oneshot::Receiver<Result<(), String>>,
        cancellation: CancellationToken,
    ) -> Self {
        Self {
            receiver,
            request_done,
            cancellation,
        }
    }

    pub async fn recv(&mut self) -> Result<Option<McpServerNotification>, String> {
        loop {
            tokio::select! {
                notification = self.receiver.recv() => match notification {
                    Ok(notification) => return Ok(Some(notification)),
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => return Ok(None),
                },
                result = &mut self.request_done => {
                    return match result {
                        Ok(result) => result.map(|_| None),
                        Err(_) => Err("MCP event stream request task ended".to_string()),
                    };
                }
            }
        }
    }
}

impl Drop for McpEventStream {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn event_stream_preserves_notifications_and_ends_after_request() {
        let (sender, _) = broadcast::channel(4);
        let (done_tx, done_rx) = oneshot::channel();
        let mut stream = McpEventStream::new(sender.subscribe(), done_rx, CancellationToken::new());
        sender
            .send(McpServerNotification {
                server_name: "server".to_string(),
                method: "notifications/events/active".to_string(),
                params: serde_json::json!({"status": "active"}),
            })
            .expect("event subscriber remains open");

        let notification = stream
            .recv()
            .await
            .expect("notification receive")
            .expect("active notification");
        assert_eq!(notification.method, "notifications/events/active");

        done_tx.send(Ok(())).expect("request completion receiver");
        assert!(stream.recv().await.expect("stream completion").is_none());
    }
}
