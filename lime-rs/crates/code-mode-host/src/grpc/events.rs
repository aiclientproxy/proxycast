//! Session event publication helpers.

use super::session::GrpcSession;
use code_mode_protocol::grpc as proto;
use code_mode_protocol::RuntimeCodeModeCellId;

pub(crate) async fn cell_closed(session: &GrpcSession, cell_id: RuntimeCodeModeCellId) {
    let (execution_id, final_tool_call_sequence) = session.take_execution(cell_id.as_str()).await;
    let _ = session
        .publish_event(proto::session_event::Event::CellClosed(proto::CellClosed {
            execution_id,
            cell_id: cell_id.to_string(),
            final_tool_call_sequence,
        }))
        .await;
}

pub(crate) async fn tool_call_cancelled(session: &GrpcSession, invocation_id: String) {
    let _ = session
        .publish_event(proto::session_event::Event::ToolCallCancelled(
            proto::ToolCallCancelled { invocation_id },
        ))
        .await;
}

pub(crate) async fn notification_cancelled(session: &GrpcSession, notification_id: String) {
    let _ = session
        .publish_event(proto::session_event::Event::NotificationCancelled(
            proto::NotificationCancelled { notification_id },
        ))
        .await;
}
