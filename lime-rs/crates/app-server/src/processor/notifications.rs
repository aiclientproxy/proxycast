use super::{
    CommandExecNotificationHook, ConnectionId, FsNotificationHook, ProcessNotificationHook,
    RequestProcessor,
};
use app_server_protocol::protocol::v2::ServerNotification;
use futures::future::BoxFuture;
use std::sync::Arc;

pub(crate) type ServerNotificationHook =
    Arc<dyn Fn(ServerNotification) -> BoxFuture<'static, ()> + Send + Sync>;
pub(crate) type ConnectionServerNotificationHook =
    Arc<dyn Fn(ConnectionId, ServerNotification) -> BoxFuture<'static, ()> + Send + Sync>;

impl RequestProcessor {
    pub(crate) fn with_server_notification_hook(mut self, hook: ServerNotificationHook) -> Self {
        self.server_notification_hook = Some(hook);
        self
    }

    pub(crate) fn with_connection_server_notification_hook(
        mut self,
        hook: ConnectionServerNotificationHook,
    ) -> Self {
        self.connection_server_notification_hook = Some(hook);
        self
    }

    pub(crate) fn with_process_notification_hook(mut self, hook: ProcessNotificationHook) -> Self {
        self.process = self.process.with_notification_hook(hook);
        self
    }

    pub(crate) fn with_command_exec_notification_hook(
        mut self,
        hook: CommandExecNotificationHook,
    ) -> Self {
        self.command_exec = self.command_exec.with_notification_hook(hook);
        self
    }

    pub(crate) fn with_fs_notification_hook(mut self, hook: FsNotificationHook) -> Self {
        self.fs = self.fs.with_notification_hook(hook);
        self
    }

    pub(crate) async fn close_process_connection(&self, connection_id: ConnectionId) {
        self.process.connection_closed(connection_id).await;
    }

    pub(crate) async fn close_command_exec_connection(&self, connection_id: ConnectionId) {
        self.command_exec.connection_closed(connection_id).await;
    }

    pub(crate) async fn close_fs_connection(&self, connection_id: ConnectionId) {
        self.fs.connection_closed(connection_id).await;
    }

    pub(crate) async fn publish_server_notification(&self, notification: ServerNotification) {
        if let Some(hook) = self.server_notification_hook.as_ref() {
            hook(notification).await;
        }
    }

    pub(super) async fn publish_connection_server_notification(
        &self,
        connection_id: ConnectionId,
        notification: ServerNotification,
    ) {
        if let Some(hook) = self.connection_server_notification_hook.as_ref() {
            hook(connection_id, notification).await;
        }
    }
}
