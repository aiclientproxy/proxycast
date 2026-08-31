//! MCP 客户端实现
//!
//! 实现 rmcp 的 ClientHandler trait，处理通知和回调。
//! 使用 DynEmitter 进行事件发射，与具体桌面宿主解耦。

#![allow(dead_code)]

use crate::active_time::ElicitationPauseState;
use crate::elicitation::{
    ElicitationOwnerGate, ElicitationOwnerGuard, ElicitationRequestRouter, ElicitationRouterError,
};
use crate::events::{McpResourceUpdatedPayload, McpResourcesUpdatedPayload, McpServerNotification};
use crate::McpRuntimeOwner;
use lime_core::DynEmitter;
use rmcp::{
    handler::client::progress::{ProgressDispatcher, ProgressSubscriber},
    model::{
        ClientCapabilities, ClientInfo, CreateElicitationRequestParam, ElicitationCapability,
        Implementation, LoggingMessageNotificationParam, ProgressNotificationParam, ProgressToken,
        ProtocolVersion, ResourceUpdatedNotificationParam,
    },
    service::NotificationContext,
    ClientHandler, RoleClient,
};
use std::sync::Arc;
use tokio::sync::broadcast;
use tool_runtime::mcp_connection::McpCallScope;
use tracing::{debug, info, warn};

/// 进度通知事件 Payload
#[derive(Debug, Clone, serde::Serialize)]
pub struct McpProgressPayload {
    pub server_name: String,
    pub progress_token: serde_json::Value,
    pub progress: f64,
    pub total: Option<f64>,
    pub message: Option<String>,
}

/// 日志消息事件 Payload
#[derive(Debug, Clone, serde::Serialize)]
pub struct McpLogMessagePayload {
    pub server_name: String,
    pub level: String,
    pub logger: Option<String>,
    pub data: serde_json::Value,
}

/// Lime MCP 客户端处理器
pub struct LimeMcpClient {
    emitter: Option<DynEmitter>,
    server_name: String,
    progress_dispatcher: ProgressDispatcher,
    elicitation_router: Option<ElicitationRequestRouter>,
    runtime_owner: Option<McpRuntimeOwner>,
    elicitation_pause_state: ElicitationPauseState,
    elicitation_owner: ElicitationOwnerGate,
    notification_sender: Option<broadcast::Sender<McpServerNotification>>,
}

impl LimeMcpClient {
    pub fn new(server_name: String, emitter: Option<DynEmitter>) -> Self {
        Self::from_parts(server_name, emitter, None, None, None)
    }

    pub fn with_elicitation_router(
        server_name: String,
        emitter: Option<DynEmitter>,
        elicitation_router: ElicitationRequestRouter,
    ) -> Self {
        Self::from_parts(server_name, emitter, Some(elicitation_router), None, None)
    }

    pub fn with_runtime_elicitation_router(
        server_name: String,
        emitter: Option<DynEmitter>,
        elicitation_router: ElicitationRequestRouter,
        runtime_owner: McpRuntimeOwner,
    ) -> Self {
        Self::with_runtime_elicitation_router_and_notifications(
            server_name,
            emitter,
            elicitation_router,
            runtime_owner,
            None,
        )
    }

    pub fn with_runtime_elicitation_router_and_notifications(
        server_name: String,
        emitter: Option<DynEmitter>,
        elicitation_router: ElicitationRequestRouter,
        runtime_owner: McpRuntimeOwner,
        notification_sender: Option<broadcast::Sender<McpServerNotification>>,
    ) -> Self {
        Self::from_parts(
            server_name,
            emitter,
            Some(elicitation_router),
            Some(runtime_owner),
            notification_sender,
        )
    }

    fn from_parts(
        server_name: String,
        emitter: Option<DynEmitter>,
        elicitation_router: Option<ElicitationRequestRouter>,
        runtime_owner: Option<McpRuntimeOwner>,
        notification_sender: Option<broadcast::Sender<McpServerNotification>>,
    ) -> Self {
        Self {
            emitter,
            server_name,
            progress_dispatcher: ProgressDispatcher::new(),
            elicitation_router,
            runtime_owner,
            elicitation_pause_state: ElicitationPauseState::new(),
            elicitation_owner: ElicitationOwnerGate::default(),
            notification_sender,
        }
    }

    pub(crate) async fn handle_form_elicitation(
        &self,
        request: CreateElicitationRequestParam,
        scope: McpCallScope,
        meta: Option<serde_json::Value>,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> Result<crate::elicitation::ElicitationResponse, ElicitationRouterError> {
        let router = self
            .elicitation_router
            .as_ref()
            .ok_or(ElicitationRouterError::NoRequestRouter)?;
        let runtime_owner = self
            .runtime_owner
            .as_ref()
            .ok_or(ElicitationRouterError::NoRequestRouter)?;
        let _pause = self.elicitation_pause_state.enter();
        router
            .request_with_scope(
                self.server_name.clone(),
                runtime_owner.clone(),
                scope,
                request,
                meta,
                cancellation,
            )
            .await
    }

    pub(crate) async fn enter_elicitation_owner(
        &self,
        scope: Option<McpCallScope>,
    ) -> ElicitationOwnerGuard {
        self.elicitation_owner.enter(scope).await
    }

    pub(crate) fn resolve_elicitation_request_meta(
        &self,
        meta: rmcp::model::Meta,
    ) -> (Option<McpCallScope>, Option<serde_json::Value>) {
        self.elicitation_owner.resolve_request_meta(meta)
    }

    pub(crate) fn elicitation_pause_state(&self) -> ElicitationPauseState {
        self.elicitation_pause_state.clone()
    }

    pub(crate) async fn subscribe_progress(
        &self,
        progress_token: ProgressToken,
    ) -> ProgressSubscriber {
        self.progress_dispatcher.subscribe(progress_token).await
    }

    /// 发送事件（通过 DynEmitter）
    fn emit_event<T: serde::Serialize>(&self, event: &str, payload: &T) {
        if let Some(ref emitter) = self.emitter {
            if let Ok(value) = serde_json::to_value(payload) {
                if let Err(e) = emitter.emit_event(event, &value) {
                    warn!(
                        server_name = %self.server_name,
                        event = %event,
                        error = %e,
                        "发送事件失败"
                    );
                }
            }
        }
    }

    fn emit_runtime_notification(&self, method: &str, params: serde_json::Value) {
        if let Some(sender) = &self.notification_sender {
            let _ = sender.send(McpServerNotification {
                server_name: self.server_name.clone(),
                method: method.to_string(),
                params,
            });
        }
    }
}

impl ClientHandler for LimeMcpClient {
    fn get_info(&self) -> ClientInfo {
        let supports_runtime_elicitation =
            self.elicitation_router.is_some() && self.runtime_owner.is_some();
        let mut capabilities = ClientCapabilities::default();
        if supports_runtime_elicitation {
            capabilities.elicitation = Some(ElicitationCapability::default());
        }
        ClientInfo {
            protocol_version: if supports_runtime_elicitation {
                ProtocolVersion::V_2025_06_18
            } else {
                ProtocolVersion::V_2025_03_26
            },
            capabilities,
            client_info: Implementation {
                name: "lime".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                icons: None,
                title: Some("Lime MCP Client".to_string()),
                website_url: Some("https://github.com/aiclientproxy/lime".to_string()),
            },
        }
    }

    async fn on_progress(
        &self,
        params: ProgressNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) {
        debug!(
            server_name = %self.server_name,
            progress_token = ?params.progress_token,
            progress = params.progress,
            total = ?params.total,
            "收到 MCP 进度通知"
        );

        self.progress_dispatcher
            .handle_notification(params.clone())
            .await;

        let payload = McpProgressPayload {
            server_name: self.server_name.clone(),
            progress_token: serde_json::to_value(&params.progress_token)
                .unwrap_or(serde_json::Value::Null),
            progress: params.progress,
            total: params.total,
            message: params.message.clone(),
        };
        self.emit_event("mcp:progress", &payload);
    }

    async fn on_logging_message(
        &self,
        params: LoggingMessageNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) {
        let level_str = format!("{:?}", params.level);
        match params.level {
            rmcp::model::LoggingLevel::Debug => {
                debug!(server_name = %self.server_name, logger = ?params.logger, data = ?params.data, "MCP 服务器日志 [DEBUG]");
            }
            rmcp::model::LoggingLevel::Info => {
                info!(server_name = %self.server_name, logger = ?params.logger, data = ?params.data, "MCP 服务器日志 [INFO]");
            }
            rmcp::model::LoggingLevel::Notice => {
                info!(server_name = %self.server_name, logger = ?params.logger, data = ?params.data, "MCP 服务器日志 [NOTICE]");
            }
            rmcp::model::LoggingLevel::Warning => {
                warn!(server_name = %self.server_name, logger = ?params.logger, data = ?params.data, "MCP 服务器日志 [WARNING]");
            }
            _ => {
                tracing::error!(server_name = %self.server_name, logger = ?params.logger, data = ?params.data, level = %level_str, "MCP 服务器日志");
            }
        }

        let payload = McpLogMessagePayload {
            server_name: self.server_name.clone(),
            level: level_str,
            logger: params.logger.clone(),
            data: params.data.clone(),
        };
        self.emit_event("mcp:log_message", &payload);
    }

    async fn on_resource_updated(
        &self,
        params: ResourceUpdatedNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) {
        debug!(
            server_name = %self.server_name,
            uri = %params.uri,
            "收到 MCP 资源更新通知"
        );

        self.emit_event(
            "mcp:resource_updated",
            &McpResourceUpdatedPayload {
                server_name: self.server_name.clone(),
                uri: params.uri.clone(),
            },
        );
    }

    async fn on_resource_list_changed(&self, _context: NotificationContext<RoleClient>) {
        debug!(server_name = %self.server_name, "收到 MCP 资源列表更新通知");

        self.emit_event(
            "mcp:resources_updated",
            &McpResourcesUpdatedPayload {
                server_name: self.server_name.clone(),
            },
        );
        self.emit_runtime_notification(
            "notifications/resources/list_changed",
            serde_json::Value::Object(Default::default()),
        );
    }

    async fn on_tool_list_changed(&self, _context: NotificationContext<RoleClient>) {
        debug!(server_name = %self.server_name, "收到 MCP 工具列表更新通知");

        self.emit_runtime_notification(
            "notifications/tools/list_changed",
            serde_json::Value::Object(Default::default()),
        );
    }

    async fn on_prompt_list_changed(&self, _context: NotificationContext<RoleClient>) {
        debug!(server_name = %self.server_name, "收到 MCP 提示词列表更新通知");

        self.emit_runtime_notification(
            "notifications/prompts/list_changed",
            serde_json::Value::Object(Default::default()),
        );
    }

    async fn on_custom_notification(
        &self,
        notification: rmcp::model::CustomNotification,
        _context: NotificationContext<RoleClient>,
    ) {
        let params = notification
            .params
            .unwrap_or(serde_json::Value::Object(Default::default()));
        self.emit_runtime_notification(&notification.method, params);
    }
}

/// MCP 客户端包装器
pub struct McpClientWrapper {
    pub server_name: String,
    pub config: super::types::McpServerConfig,
    pub server_info: Option<super::types::McpServerCapabilities>,
    pub running_service: Option<
        Arc<
            rmcp::service::RunningService<
                rmcp::RoleClient,
                crate::client_service::LimeMcpClientService,
            >,
        >,
    >,
    stdio_process: Option<crate::stdio_process::StdioProcessHandle>,
    stderr_task: Option<tokio::task::JoinHandle<()>>,
}

impl McpClientWrapper {
    pub fn new(
        server_name: String,
        config: super::types::McpServerConfig,
        _emitter: Option<DynEmitter>,
    ) -> Self {
        Self {
            server_name,
            config,
            server_info: None,
            running_service: None,
            stdio_process: None,
            stderr_task: None,
        }
    }

    pub(crate) fn set_stdio_lifecycle(
        &mut self,
        process: crate::stdio_process::StdioProcessHandle,
        stderr_task: Option<tokio::task::JoinHandle<()>>,
    ) {
        self.stdio_process = Some(process);
        self.stderr_task = stderr_task;
    }

    pub fn set_server_info(&mut self, info: super::types::McpServerCapabilities) {
        self.server_info = Some(info);
    }

    pub fn set_running_service(
        &mut self,
        service: rmcp::service::RunningService<
            rmcp::RoleClient,
            crate::client_service::LimeMcpClientService,
        >,
    ) {
        self.running_service = Some(Arc::new(service));
    }

    pub fn running_service(
        &self,
    ) -> Option<
        &Arc<
            rmcp::service::RunningService<
                rmcp::RoleClient,
                crate::client_service::LimeMcpClientService,
            >,
        >,
    > {
        self.running_service.as_ref()
    }

    pub fn running_service_arc(
        &self,
    ) -> Option<
        Arc<
            rmcp::service::RunningService<
                rmcp::RoleClient,
                crate::client_service::LimeMcpClientService,
            >,
        >,
    > {
        self.running_service.clone()
    }

    pub fn shutdown(&mut self) {
        if let Some(service) = &self.running_service {
            service.cancellation_token().cancel();
        }
        if let Some(process) = self.stdio_process.take() {
            process.terminate();
        }
        if let Some(stderr_task) = self.stderr_task.take() {
            stderr_task.abort();
        }
        self.running_service = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use rmcp::model::NumberOrString;
    use std::time::Duration;

    #[tokio::test]
    async fn active_time_real_handler_pauses_for_the_full_router_wait() {
        let router = ElicitationRequestRouter::default();
        let mut requests = router.subscribe().expect("request consumer");
        let client = Arc::new(LimeMcpClient::with_runtime_elicitation_router(
            "pause-server".to_string(),
            None,
            router.clone(),
            McpRuntimeOwner {
                session_id: "session-1".to_string(),
                thread_id: "thread-1".to_string(),
            },
        ));
        let mut paused = client.elicitation_pause_state().subscribe();
        let request_client = Arc::clone(&client);

        let waiter = tokio::spawn(async move {
            request_client
                .handle_form_elicitation(
                    CreateElicitationRequestParam {
                        message: "Confirm".to_string(),
                        requested_schema: rmcp::model::ElicitationSchema::builder()
                            .build()
                            .expect("empty object schema"),
                    },
                    tool_runtime::mcp_connection::McpCallScope::new(Some("turn-1"))
                        .expect("turn correlation"),
                    None,
                    tokio_util::sync::CancellationToken::new(),
                )
                .await
        });

        let request = requests.recv().await.expect("routed elicitation");
        paused.changed().await.expect("pause state remains open");
        assert!(*paused.borrow_and_update());

        router
            .resolve(&request.id, crate::elicitation::ElicitationResponse::Cancel)
            .await
            .expect("resolve exact waiter");
        waiter
            .await
            .expect("handler task")
            .expect("router response");
        paused.changed().await.expect("pause state remains open");
        assert!(!*paused.borrow_and_update());
    }

    #[test]
    fn test_management_client_info_does_not_advertise_unimplemented_capabilities() {
        let client = LimeMcpClient::new("test-server".to_string(), None);
        let info = client.get_info();

        assert_eq!(info.client_info.name, "lime");
        assert_eq!(info.client_info.title, Some("Lime MCP Client".to_string()));
        assert_eq!(info.protocol_version, ProtocolVersion::V_2025_03_26);
        assert!(info.capabilities.sampling.is_none());
        assert!(info.capabilities.elicitation.is_none());
    }

    #[test]
    fn test_runtime_client_info_advertises_form_elicitation() {
        let client = LimeMcpClient::with_runtime_elicitation_router(
            "test-server".to_string(),
            None,
            ElicitationRequestRouter::default(),
            McpRuntimeOwner {
                session_id: "session-1".to_string(),
                thread_id: "thread-1".to_string(),
            },
        );
        let info = client.get_info();

        assert_eq!(info.protocol_version, ProtocolVersion::V_2025_06_18);
        assert_eq!(
            serde_json::to_value(&info.capabilities).expect("serialize client capabilities"),
            serde_json::json!({ "elicitation": {} })
        );
        assert!(info.capabilities.sampling.is_none());
    }

    #[test]
    fn test_management_router_without_runtime_owner_does_not_advertise_elicitation() {
        let client = LimeMcpClient::with_elicitation_router(
            "test-server".to_string(),
            None,
            ElicitationRequestRouter::default(),
        );

        let info = client.get_info();
        assert_eq!(info.protocol_version, ProtocolVersion::V_2025_03_26);
        assert!(info.capabilities.elicitation.is_none());
    }

    #[test]
    fn test_client_wrapper_creation() {
        let config = super::super::types::McpServerConfig {
            transport: super::super::types::McpServerTransport::Stdio {
                command: "test-command".to_string(),
                args: vec!["--arg1".to_string()],
                env: std::collections::HashMap::new(),
                cwd: None,
            },
            environment_id: super::super::types::DEFAULT_MCP_SERVER_ENVIRONMENT_ID.to_string(),
            enabled: true,
            startup_timeout: 30,
            tool_timeout: None,
            enabled_tools: None,
            disabled_tools: Vec::new(),
            required: false,
            supports_parallel_tool_calls: false,
            scopes: None,
            oauth: None,
            oauth_resource: None,
        };

        let wrapper = McpClientWrapper::new("test-server".to_string(), config, None);

        assert_eq!(wrapper.server_name, "test-server");
        assert_eq!(wrapper.config.command(), "test-command");
        assert!(wrapper.server_info.is_none());
        assert!(wrapper.running_service.is_none());
        assert!(wrapper.stdio_process.is_none());
        assert!(wrapper.stderr_task.is_none());
    }

    #[tokio::test]
    async fn progress_dispatcher_isolates_subscribers_by_token() {
        let client = LimeMcpClient::new("test-server".to_string(), None);
        let token_a = ProgressToken(NumberOrString::String("call-a".into()));
        let token_b = ProgressToken(NumberOrString::String("call-b".into()));
        let mut progress_a = client.subscribe_progress(token_a.clone()).await;
        let mut progress_b = client.subscribe_progress(token_b.clone()).await;

        client
            .progress_dispatcher
            .handle_notification(ProgressNotificationParam {
                progress_token: token_b.clone(),
                progress: 1.0,
                total: Some(2.0),
                message: Some("progress-b".to_string()),
            })
            .await;

        let notification_b = progress_b.next().await.expect("progress for token B");
        assert_eq!(notification_b.progress_token, token_b);
        assert_eq!(notification_b.message.as_deref(), Some("progress-b"));
        assert!(
            tokio::time::timeout(Duration::from_millis(20), progress_a.next())
                .await
                .is_err()
        );

        client
            .progress_dispatcher
            .handle_notification(ProgressNotificationParam {
                progress_token: token_a.clone(),
                progress: 2.0,
                total: Some(2.0),
                message: Some("progress-a".to_string()),
            })
            .await;

        let notification_a = progress_a.next().await.expect("progress for token A");
        assert_eq!(notification_a.progress_token, token_a);
        assert_eq!(notification_a.message.as_deref(), Some("progress-a"));
    }
}
