use super::super::{event_notifications, v2_notifications::V2NotificationProjector, RpcDispatch};
use crate::AppServerError;
use app_server_protocol::{
    JsonRpcError, JsonRpcErrorResponse, JsonRpcMessage, JsonRpcResponse, RequestId,
};

pub(super) fn into_messages(
    id: RequestId,
    result: Result<RpcDispatch, JsonRpcError>,
) -> Result<Vec<JsonRpcMessage>, AppServerError> {
    match result {
        Ok(dispatch) => {
            let mut messages =
                Vec::with_capacity(dispatch.events.len() + dispatch.notifications.len() + 1);
            messages.push(JsonRpcMessage::Response(JsonRpcResponse {
                id,
                result: dispatch.result,
            }));
            let mut event_projector = V2NotificationProjector::default();
            for event in dispatch.events {
                messages.extend(event_notifications(&mut event_projector, event)?);
            }
            for notification in dispatch.notifications {
                messages.push(JsonRpcMessage::Notification(notification));
            }
            Ok(messages)
        }
        Err(error) => Ok(vec![JsonRpcMessage::Error(JsonRpcErrorResponse {
            id,
            error,
        })]),
    }
}
