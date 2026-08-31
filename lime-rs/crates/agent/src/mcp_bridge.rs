//! MCP 桥接运行时边界
//!
//! 将 Agent reply-loop 的 MCP 调用绑定到其 Session-owned connection generation。

use futures::{stream, StreamExt};
use lime_mcp::{
    build_runtime_extension_surface, runtime_extension_name,
    McpBridgeClient as RuntimeMcpBridgeClient, McpBridgeSnapshot, McpServerNotification,
};
use rmcp::model::{
    ErrorCode, ErrorData, Extensions, JsonObject, ListToolsResult, ProgressNotification,
    ProgressNotificationMethod, PromptListChangedNotification, ResourceListChangedNotification,
    ServerNotification, ToolListChangedNotification,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tokio_util::sync::CancellationToken;
use tool_runtime::mcp_connection::McpConnectionRegistry;
use tool_runtime::mcp_connection::{
    McpCallScope, McpConnection, McpConnectionCall, McpConnectionError, McpConnectionProvenance,
};
use tool_runtime::tool_extension::{RuntimeExtensionRegistration, RuntimeExtensionSyncPlan};

pub(crate) struct McpBridgeRuntimeRegistry {
    registered_bridge_names: RwLock<HashSet<String>>,
}

impl McpBridgeRuntimeRegistry {
    pub(crate) fn new() -> Self {
        Self {
            registered_bridge_names: RwLock::new(HashSet::new()),
        }
    }

    pub(crate) async fn sync(
        &self,
        connections: &McpConnectionRegistry,
        snapshots: Vec<McpBridgeSnapshot>,
    ) -> usize {
        let mut snapshots_by_bridge_name = HashMap::new();
        let mut registrations = Vec::new();
        for snapshot in snapshots {
            let extension_name = runtime_extension_name(&snapshot.server_name);
            let surface = build_runtime_extension_surface(
                &extension_name,
                snapshot.description.clone(),
                &snapshot.tools,
            );

            let bridge_name = surface.name.clone();
            registrations.push(RuntimeExtensionRegistration::new(
                surface,
                Some(snapshot.server_name.clone()),
            ));
            snapshots_by_bridge_name.insert(bridge_name, snapshot);
        }

        let previous_bridge_names = self.registered_bridge_names.read().await.clone();
        let plan = RuntimeExtensionSyncPlan::from_registrations(
            previous_bridge_names.iter().cloned(),
            registrations,
        );
        let active_bridge_names = plan.active_names();

        for registration in &plan.registrations {
            let bridge_name = registration.config.name.clone();
            let Some(snapshot) = snapshots_by_bridge_name.get(&bridge_name) else {
                continue;
            };
            let client: Arc<Mutex<Box<dyn McpConnection>>> =
                Arc::new(Mutex::new(Box::new(McpBridgeClient::new(
                    Arc::clone(&snapshot.manager),
                    Arc::clone(&snapshot.running_service),
                    snapshot.server_name.clone(),
                    snapshot.tool_timeout,
                ))));
            let surface = registration.config.clone();

            connections
                .register(
                    bridge_name.clone(),
                    surface,
                    McpConnectionProvenance::new(
                        snapshot.environment_id.clone(),
                        snapshot.auth_scopes.clone(),
                    )
                    .with_server_name(Some(snapshot.server_name.clone()))
                    .with_plugin_id(snapshot.plugin_id.clone()),
                    snapshot.supports_parallel_tool_calls,
                    client,
                )
                .await;
        }

        for stale_name in &plan.stale_names {
            if !connections.remove(stale_name).await {
                tracing::warn!(
                    extension_name = %stale_name,
                    "[AgentRuntime] 清理过期 MCP bridge 失败"
                );
            }
        }

        let bridge_count = plan.registrations.len();
        *self.registered_bridge_names.write().await = active_bridge_names;
        bridge_count
    }
}

struct McpBridgeClient {
    manager: Arc<lime_mcp::McpClientManager>,
    inner: RuntimeMcpBridgeClient,
    server_name: String,
}

impl McpBridgeClient {
    fn new(
        manager: Arc<lime_mcp::McpClientManager>,
        service: Arc<
            rmcp::service::RunningService<rmcp::RoleClient, lime_mcp::LimeMcpClientService>,
        >,
        server_name: String,
        tool_timeout: std::time::Duration,
    ) -> Self {
        Self {
            manager,
            inner: RuntimeMcpBridgeClient::new(service, tool_timeout),
            server_name,
        }
    }

    fn request_extensions(&self) -> Extensions {
        Extensions::default()
    }
}

fn map_mcp_result<T>(
    result: Result<T, rmcp::service::ServiceError>,
) -> Result<T, McpConnectionError> {
    result
}

fn service_error_data(error: rmcp::service::ServiceError) -> ErrorData {
    match error {
        rmcp::service::ServiceError::McpError(error) => error,
        error => ErrorData::new(ErrorCode::INTERNAL_ERROR, error.to_string(), None),
    }
}

#[async_trait::async_trait]
impl McpConnection for McpBridgeClient {
    async fn list_tools(
        &self,
        cursor: Option<String>,
        cancel_token: CancellationToken,
    ) -> Result<ListToolsResult, McpConnectionError> {
        map_mcp_result(
            self.inner
                .list_tools(cursor, self.request_extensions(), cancel_token)
                .await,
        )
    }

    async fn start_call_tool(
        &self,
        name: &str,
        arguments: Option<JsonObject>,
        scope: &McpCallScope,
        cancel_token: CancellationToken,
    ) -> Result<McpConnectionCall, McpConnectionError> {
        let runtime_notifications = self.manager.subscribe_server_notifications();
        let call = self
            .inner
            .start_tool_call(
                name,
                arguments,
                self.request_extensions(),
                Some(scope),
                cancel_token,
            )
            .await?;
        let response = call.response;
        let notifications = call.progress.map(|params| {
            ServerNotification::ProgressNotification(ProgressNotification {
                params,
                method: ProgressNotificationMethod,
                extensions: Default::default(),
            })
        });
        let server_name = self.server_name.clone();
        let list_changed_notifications =
            stream::unfold(runtime_notifications, |mut receiver| async move {
                loop {
                    match receiver.recv().await {
                        Ok(notification) => return Some((notification, receiver)),
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => return None,
                    }
                }
            })
            .filter_map(move |notification| {
                let server_name = server_name.clone();
                async move {
                    (notification.server_name == server_name)
                        .then(|| map_runtime_mcp_notification(&notification))
                        .flatten()
                }
            });
        Ok(McpConnectionCall {
            response: Box::pin(async move { response.await.map_err(service_error_data) }),
            notifications: Box::pin(stream::select(notifications, list_changed_notifications)),
        })
    }
}

fn map_runtime_mcp_notification(
    notification: &McpServerNotification,
) -> Option<ServerNotification> {
    match notification.method.as_str() {
        "notifications/resources/list_changed" => Some(
            ServerNotification::ResourceListChangedNotification(ResourceListChangedNotification {
                method: Default::default(),
                extensions: Default::default(),
            }),
        ),
        "notifications/tools/list_changed" => Some(
            ServerNotification::ToolListChangedNotification(ToolListChangedNotification {
                method: Default::default(),
                extensions: Default::default(),
            }),
        ),
        "notifications/prompts/list_changed" => Some(
            ServerNotification::PromptListChangedNotification(PromptListChangedNotification {
                method: Default::default(),
                extensions: Default::default(),
            }),
        ),
        _ => None,
    }
}
