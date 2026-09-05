//! Tool subscription matching and callback routing.

use super::{events, session::GrpcSession};
use code_mode_protocol::grpc as proto;
use code_mode_protocol::RuntimeCodeModeNestedToolCall;
use serde_json::Value;
use std::sync::atomic::Ordering;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

const MAX_PENDING_DELEGATE_CALLS: usize = code_mode_protocol::host::MAX_PENDING_DELEGATE_CALLS;

pub(crate) async fn publish_tool_call(
    session: &GrpcSession,
    invocation: RuntimeCodeModeNestedToolCall,
    cancellation: CancellationToken,
) -> Result<Value, String> {
    let invocation_id = uuid::Uuid::new_v4().to_string();
    let (sender, receiver) = oneshot::channel();
    let tool_name = invocation.tool_name.clone();
    let execution_id = session
        .execution_id_for_cell(invocation.cell_id.as_str())
        .await
        .ok_or_else(|| "code-mode cell has no owning execution".to_string())?;
    let sequence = session
        .next_tool_call_sequence(invocation.cell_id.as_str())
        .await?;
    let mut pending = session.pending.lock().await;
    if pending.len() >= MAX_PENDING_DELEGATE_CALLS {
        return Err("code-mode delegate request limit exceeded".to_string());
    }
    pending.insert(invocation_id.clone(), sender);
    drop(pending);
    let message = proto::ToolCall {
        session_id: session.id.clone(),
        execution_id,
        cell_id: invocation.cell_id.to_string(),
        invocation_id: invocation_id.clone(),
        runtime_tool_call_id: invocation.runtime_tool_call_id,
        tool_name: Some(proto::ToolName {
            name: tool_name.clone(),
            namespace: None,
        }),
        tool_kind: match invocation.kind {
            code_mode_protocol::CodeModeToolKind::Function => proto::ToolKind::Function as i32,
            code_mode_protocol::CodeModeToolKind::Freeform => proto::ToolKind::Freeform as i32,
        },
        input_json: invocation
            .input
            .as_ref()
            .map(serde_json::to_vec)
            .transpose()
            .map_err(|error| error.to_string())?,
        sequence,
    };
    let subscribers = {
        let mut subscribers = session.subscribers.lock().await;
        subscribers.retain(|subscriber| !subscriber.sender.is_closed());
        subscribers
            .iter()
            .filter(|subscriber| {
                subscriber.tool_names.is_empty()
                    || subscriber
                        .tool_names
                        .iter()
                        .any(|name| name.name == tool_name && name.namespace.is_none())
            })
            .map(|subscriber| subscriber.sender.clone())
            .collect::<Vec<_>>()
    };
    if subscribers.is_empty() {
        session.pending.lock().await.remove(&invocation_id);
        return Err("no active code-mode tool subscription".to_string());
    }
    let start = session.next_subscriber.fetch_add(1, Ordering::Relaxed) as usize;
    let mut delivered = false;
    for offset in 0..subscribers.len() {
        let index = (start + offset) % subscribers.len();
        if subscribers[index].send(Ok(message.clone())).await.is_ok() {
            delivered = true;
            break;
        }
    }
    if !delivered {
        session.pending.lock().await.remove(&invocation_id);
        return Err("no matching code-mode tool subscription".to_string());
    }
    tokio::select! {
        result = receiver => result.map_err(|_| "code-mode tool completion channel closed".to_string())?,
        _ = cancellation.cancelled() => {
            session.pending.lock().await.remove(&invocation_id);
            events::tool_call_cancelled(session, invocation_id).await;
            Err("code-mode tool invocation cancelled".to_string())
        }
    }
}
