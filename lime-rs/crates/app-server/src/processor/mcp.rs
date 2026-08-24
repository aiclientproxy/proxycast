//! mcp domain handlers for the App Server processor.

use super::ConnectionRequestId;
use super::{dispatch_result, parse_params, to_jsonrpc_error, RequestProcessor, RpcDispatch};
use app_server_protocol::protocol::v2::{
    ListMcpServerStatusParams, McpServerEventNotification, McpServerEventStreamNotification,
    McpServerEventStreamStartParams, McpServerEventStreamStartResponse,
    McpServerEventStreamStopParams, McpServerEventStreamStopResponse,
    McpServerOauthLoginCompletedNotification,
    McpServerResourceReadParams as V2McpServerResourceReadParams, McpServerStartupState,
    McpServerStatusUpdatedNotification, McpServerToolCallParams as V2McpServerToolCallParams,
    ServerNotification,
};
use app_server_protocol::{
    JsonRpcError, McpPromptGetParams, McpResourceSubscribeParams, McpResourceUnsubscribeParams,
    McpServerCreateParams, McpServerDeleteParams, McpServerEnabledSetParams,
    McpServerImportFromAppParams, McpServerOauthLoginParams, McpServerOauthLoginResponse,
    McpServerStartParams, McpServerStopParams, McpServerUpdateParams, McpToolListForContextParams,
    McpToolSearchParams,
};
use std::time::Duration;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

pub(crate) const MCP_EVENT_STREAM_STARTUP_TIMEOUT: Duration = Duration::from_secs(90);
pub(crate) const MCP_EVENT_STREAM_RECONNECT_DELAY: Duration = Duration::from_secs(1);
pub(crate) const MCP_EVENT_STREAM_MAX_RECONNECT_ATTEMPTS: u32 = 3;
const MCP_EVENT_STREAM_PREACTIVE_BUFFER_LIMIT: usize = 256;

pub(crate) struct McpEventStreamTask {
    pub(crate) thread_id: String,
    pub(crate) task: JoinHandle<()>,
}

impl RequestProcessor {
    pub(crate) async fn close_mcp_event_streams(
        &self,
        connection_id: app_server_transport::ConnectionId,
    ) {
        let tasks = {
            let mut streams = self
                .mcp_event_streams
                .lock()
                .expect("MCP event stream mutex poisoned");
            streams
                .extract_if(|(owner, _), _| *owner == connection_id)
                .map(|(_, task)| task.task)
                .collect::<Vec<_>>()
        };
        for task in tasks {
            task.abort();
            let _ = task.await;
        }
    }

    pub(crate) async fn close_mcp_event_stream(
        &self,
        connection_id: app_server_transport::ConnectionId,
        subscription_id: &str,
    ) {
        let task = self
            .mcp_event_streams
            .lock()
            .expect("MCP event stream mutex poisoned")
            .remove(&(connection_id, subscription_id.to_string()));
        if let Some(task) = task {
            task.task.abort();
            let _ = task.task.await;
        }
    }

    pub(crate) async fn close_mcp_event_streams_for_thread(&self, thread_id: &str) {
        let tasks = {
            let mut streams = self
                .mcp_event_streams
                .lock()
                .expect("MCP event stream mutex poisoned");
            streams
                .extract_if(|_, task| task.thread_id == thread_id)
                .map(|((connection_id, subscription_id), task)| {
                    (connection_id, subscription_id, task.task)
                })
                .collect::<Vec<_>>()
        };
        for (connection_id, subscription_id, task) in tasks {
            self.publish_connection_server_notification(
                connection_id,
                ServerNotification::McpServerEventStream(McpServerEventStreamNotification {
                    subscription_id,
                    notification: McpServerEventNotification {
                        method: "notifications/events/terminated".to_string(),
                        params: serde_json::Value::Object(Default::default()),
                    },
                }),
            )
            .await;
            task.abort();
            let _ = task.await;
        }
    }

    pub(super) async fn handle_mcp_server_list_impl(&self) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let response = self
            .runtime
            .list_mcp_servers()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_server_status_list_v2_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ListMcpServerStatusParams = parse_params(params)?;
        let response = self
            .runtime
            .list_mcp_servers_with_status_v2(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_server_create_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpServerCreateParams = parse_params(params)?;
        let response = self
            .runtime
            .create_mcp_server(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_server_update_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpServerUpdateParams = parse_params(params)?;
        let response = self
            .runtime
            .update_mcp_server(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_server_delete_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpServerDeleteParams = parse_params(params)?;
        let response = self
            .runtime
            .delete_mcp_server(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_server_enabled_set_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpServerEnabledSetParams = parse_params(params)?;
        let response = self
            .runtime
            .set_mcp_server_enabled(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_server_import_from_app_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpServerImportFromAppParams = parse_params(params)?;
        let response = self
            .runtime
            .import_mcp_servers_from_app(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_server_sync_all_to_live_impl(
        &self,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let response = self
            .runtime
            .sync_all_mcp_servers_to_live()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_server_start_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpServerStartParams = parse_params(params)?;
        let server_name = params.name.clone();
        self.publish_server_notification(ServerNotification::McpServerStatusUpdated(
            McpServerStatusUpdatedNotification {
                thread_id: None,
                name: server_name.clone(),
                status: McpServerStartupState::Starting,
                error: None,
                failure_reason: None,
            },
        ))
        .await;

        match self.runtime.start_mcp_server(params).await {
            Ok(response) => {
                self.publish_server_notification(ServerNotification::McpServerStatusUpdated(
                    McpServerStatusUpdatedNotification {
                        thread_id: None,
                        name: server_name,
                        status: McpServerStartupState::Ready,
                        error: None,
                        failure_reason: None,
                    },
                ))
                .await;
                dispatch_result(response)
            }
            Err(error) => {
                let error_message = error.to_string();
                self.publish_server_notification(ServerNotification::McpServerStatusUpdated(
                    McpServerStatusUpdatedNotification {
                        thread_id: None,
                        name: server_name,
                        status: McpServerStartupState::Failed,
                        error: Some(error_message),
                        failure_reason: None,
                    },
                ))
                .await;
                Err(to_jsonrpc_error(error))
            }
        }
    }

    pub(super) async fn handle_mcp_server_stop_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpServerStopParams = parse_params(params)?;
        let response = self
            .runtime
            .stop_mcp_server(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_server_oauth_login_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpServerOauthLoginParams = parse_params(params)?;
        let server_name = params.name.clone();
        let handle = self
            .runtime
            .login_mcp_server_oauth(params)
            .await
            .map_err(to_jsonrpc_error)?;
        let response = McpServerOauthLoginResponse {
            authorization_url: handle.authorization_url.clone(),
            state: handle.state.clone(),
        };
        let processor = self.clone();
        tokio::spawn(async move {
            let (success, error) = match handle.wait().await {
                Ok(()) => (true, None),
                Err(error) => (false, Some(error.to_string())),
            };
            processor
                .publish_server_notification(ServerNotification::McpServerOauthLoginCompleted(
                    McpServerOauthLoginCompletedNotification {
                        name: server_name,
                        thread_id: None,
                        success,
                        error,
                    },
                ))
                .await;
        });
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_tool_list_impl(&self) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let response = self
            .runtime
            .list_mcp_tools()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_tool_list_for_context_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpToolListForContextParams = parse_params(params)?;
        let response = self
            .runtime
            .list_mcp_tools_for_context(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_tool_search_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpToolSearchParams = parse_params(params)?;
        let response = self
            .runtime
            .search_mcp_tools(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_server_resource_read_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: V2McpServerResourceReadParams = parse_params(params)?;
        let response = self
            .runtime
            .read_mcp_server_resource(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_server_tool_call_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: V2McpServerToolCallParams = parse_params(params)?;
        let response = self
            .runtime
            .call_mcp_server_tool(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_server_event_stream_start_impl(
        &self,
        params: Option<serde_json::Value>,
        connection_request_id: Option<ConnectionRequestId>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let connection_id = connection_request_id
            .map(|request| request.connection_id)
            .ok_or_else(|| {
                JsonRpcError::new(
                    app_server_protocol::error_codes::INVALID_REQUEST,
                    "MCP event streams require a transport connection",
                )
            })?;
        let params: McpServerEventStreamStartParams = parse_params(params)?;
        if params.thread_id.trim().is_empty()
            || params.server.trim().is_empty()
            || params.subscription_id.trim().is_empty()
            || params.name.trim().is_empty()
        {
            return Err(JsonRpcError::new(
                app_server_protocol::error_codes::INVALID_PARAMS,
                "MCP event stream start requires threadId, server, subscriptionId, and name",
            ));
        }
        if !self
            .thread_states
            .subscribed_connection_ids(&agent_protocol::ThreadId::new(params.thread_id.clone()))
            .await
            .contains(&connection_id)
        {
            return Err(JsonRpcError::new(
                app_server_protocol::error_codes::INVALID_REQUEST,
                format!(
                    "connection is not subscribed to thread '{}'",
                    params.thread_id
                ),
            ));
        }
        if !self
            .runtime
            .has_mcp_server_for_thread(&params.thread_id, &params.server)
            .await
            .map_err(to_jsonrpc_error)?
        {
            return Err(JsonRpcError::new(
                app_server_protocol::error_codes::INVALID_REQUEST,
                format!(
                    "MCP server '{}' is not running for this thread",
                    params.server
                ),
            ));
        }
        let key = (connection_id, params.subscription_id.clone());
        {
            let streams = self
                .mcp_event_streams
                .lock()
                .expect("MCP event stream mutex poisoned");
            if matches!(streams.get(&key), Some(_)) {
                return Err(JsonRpcError::new(
                    app_server_protocol::error_codes::INVALID_REQUEST,
                    format!(
                        "MCP event subscription '{}' already exists",
                        params.subscription_id
                    ),
                ));
            }
        }
        let stream = self
            .runtime
            .open_mcp_server_event_stream(
                &params.thread_id,
                &params.server,
                &params.name,
                params.arguments.clone(),
                params.meta.clone(),
            )
            .await
            .map_err(to_jsonrpc_error)?;
        let processor = self.clone();
        let subscription_id = params.subscription_id.clone();
        let task_subscription_id = subscription_id.clone();
        let server_name = params.server.clone();
        let thread_id = params.thread_id.clone();
        let task_thread_id = thread_id.clone();
        let (ready_tx, ready_rx) = oneshot::channel();
        let task = tokio::spawn(async move {
            let mut ready_tx = Some(ready_tx);
            let mut stream = stream;
            let mut reconnect_attempts = 0;
            let mut reconnect_deadline = None;
            let mut activated = false;
            let mut terminated_seen = false;
            let mut preactive_notifications = Vec::new();
            loop {
                let notification = match stream.recv().await {
                    Ok(Some(notification)) if notification.server_name == server_name => {
                        notification
                    }
                    Ok(Some(_)) => continue,
                    // A replaced runtime cancels the old request, which is
                    // surfaced as `Err` by `McpEventStream`; after the
                    // subscription is active that is recoverable just like
                    // a clean stream end. Startup errors remain fail-closed
                    // in the arms below.
                    Ok(None) | Err(_) if ready_tx.is_none() => {
                        if reconnect_attempts >= MCP_EVENT_STREAM_MAX_RECONNECT_ATTEMPTS {
                            break;
                        }
                        reconnect_attempts += 1;
                        let deadline = *reconnect_deadline.get_or_insert_with(|| {
                            tokio::time::Instant::now() + MCP_EVENT_STREAM_STARTUP_TIMEOUT
                        });
                        let remaining =
                            deadline.saturating_duration_since(tokio::time::Instant::now());
                        if remaining.is_zero() {
                            break;
                        }
                        tokio::time::sleep(
                            MCP_EVENT_STREAM_RECONNECT_DELAY * (1 << (reconnect_attempts - 1)),
                        )
                        .await;
                        if tokio::time::Instant::now() > deadline {
                            break;
                        }
                        match processor
                            .runtime
                            .open_mcp_server_event_stream(
                                &task_thread_id,
                                &server_name,
                                &params.name,
                                params.arguments.clone(),
                                params.meta.clone(),
                            )
                            .await
                        {
                            Ok(reconnected) => {
                                stream = reconnected;
                                continue;
                            }
                            Err(error) => {
                                tracing::debug!(
                                    %connection_id,
                                    %task_thread_id,
                                    %task_subscription_id,
                                    %error,
                                    "MCP event stream reconnect attempt failed"
                                );
                                continue;
                            }
                        }
                    }
                    Ok(None) => {
                        if let Some(ready_tx) = ready_tx.take() {
                            let _ = ready_tx.send(Err(
                                "MCP event stream ended before becoming active".to_string(),
                            ));
                        }
                        break;
                    }
                    Err(error) => {
                        if let Some(ready_tx) = ready_tx.take() {
                            let _ = ready_tx.send(Err(error));
                        }
                        break;
                    }
                };

                let active = notification.method == "notifications/events/active";
                let terminated = notification.method == "notifications/events/terminated";
                if !activated && !active {
                    if terminated {
                        terminated_seen = true;
                        if let Some(ready_tx) = ready_tx.take() {
                            let _ = ready_tx.send(Err(
                                "MCP event stream ended before becoming active".to_string(),
                            ));
                        }
                        break;
                    }
                    if preactive_notifications.len() >= MCP_EVENT_STREAM_PREACTIVE_BUFFER_LIMIT {
                        if let Some(ready_tx) = ready_tx.take() {
                            let _ = ready_tx.send(Err(
                                "MCP event stream exceeded its pre-active notification limit"
                                    .to_string(),
                            ));
                        }
                        break;
                    }
                    preactive_notifications.push(notification);
                    continue;
                }
                let payload = McpServerEventStreamNotification {
                    subscription_id: task_subscription_id.clone(),
                    notification: McpServerEventNotification {
                        method: notification.method,
                        params: notification.params,
                    },
                };
                processor
                    .publish_connection_server_notification(
                        connection_id,
                        ServerNotification::McpServerEventStream(payload),
                    )
                    .await;
                if active {
                    activated = true;
                    if let Some(ready_tx) = ready_tx.take() {
                        let _ = ready_tx.send(Ok(()));
                    }
                    for pending in preactive_notifications.drain(..) {
                        processor
                            .publish_connection_server_notification(
                                connection_id,
                                ServerNotification::McpServerEventStream(
                                    McpServerEventStreamNotification {
                                        subscription_id: task_subscription_id.clone(),
                                        notification: McpServerEventNotification {
                                            method: pending.method,
                                            params: pending.params,
                                        },
                                    },
                                ),
                            )
                            .await;
                    }
                }
                if terminated {
                    terminated_seen = true;
                    break;
                }
                if !active {
                    reconnect_attempts = 0;
                    reconnect_deadline = None;
                }
            }
            if activated && !terminated_seen {
                processor
                    .publish_connection_server_notification(
                        connection_id,
                        ServerNotification::McpServerEventStream(
                            McpServerEventStreamNotification {
                                subscription_id: task_subscription_id.clone(),
                                notification: McpServerEventNotification {
                                    method: "notifications/events/terminated".to_string(),
                                    params: serde_json::Value::Object(Default::default()),
                                },
                            },
                        ),
                    )
                    .await;
            }
            tracing::debug!(%connection_id, %task_thread_id, %task_subscription_id, "MCP event stream ended");
            processor
                .mcp_event_streams
                .lock()
                .expect("MCP event stream mutex poisoned")
                .remove(&(connection_id, task_subscription_id));
        });
        {
            let mut streams = self
                .mcp_event_streams
                .lock()
                .expect("MCP event stream mutex poisoned");
            if matches!(streams.get(&key), Some(_)) {
                return Err(JsonRpcError::new(
                    app_server_protocol::error_codes::INVALID_REQUEST,
                    format!(
                        "MCP event subscription '{}' already exists",
                        params.subscription_id
                    ),
                ));
            }
            streams.insert(
                key,
                McpEventStreamTask {
                    thread_id: thread_id.clone(),
                    task,
                },
            );
        }
        match tokio::time::timeout(MCP_EVENT_STREAM_STARTUP_TIMEOUT, ready_rx).await {
            Ok(Ok(Ok(()))) => dispatch_result(McpServerEventStreamStartResponse {}),
            Ok(Ok(Err(error))) => {
                self.close_mcp_event_stream(connection_id, &subscription_id)
                    .await;
                Err(JsonRpcError::new(
                    app_server_protocol::error_codes::RUNTIME_ERROR,
                    error,
                ))
            }
            Ok(Err(_)) => {
                self.close_mcp_event_stream(connection_id, &subscription_id)
                    .await;
                Err(JsonRpcError::new(
                    app_server_protocol::error_codes::RUNTIME_ERROR,
                    "MCP event stream ended before becoming active",
                ))
            }
            Err(_) => {
                self.close_mcp_event_stream(connection_id, &subscription_id)
                    .await;
                Err(JsonRpcError::new(
                    app_server_protocol::error_codes::RUNTIME_ERROR,
                    "MCP event stream startup timed out",
                ))
            }
        }
    }

    pub(super) async fn handle_mcp_server_event_stream_stop_impl(
        &self,
        params: Option<serde_json::Value>,
        connection_request_id: Option<ConnectionRequestId>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let connection_id = connection_request_id
            .map(|request| request.connection_id)
            .ok_or_else(|| {
                JsonRpcError::new(
                    app_server_protocol::error_codes::INVALID_REQUEST,
                    "MCP event streams require a transport connection",
                )
            })?;
        let params: McpServerEventStreamStopParams = parse_params(params)?;
        let key = (connection_id, params.subscription_id.clone());
        let task = self
            .mcp_event_streams
            .lock()
            .expect("MCP event stream mutex poisoned")
            .remove(&key);
        let Some(task) = task else {
            return Err(JsonRpcError::new(
                app_server_protocol::error_codes::INVALID_REQUEST,
                format!(
                    "MCP event subscription '{}' does not exist",
                    params.subscription_id
                ),
            ));
        };
        task.task.abort();
        let _ = task.task.await;
        dispatch_result(McpServerEventStreamStopResponse {})
    }

    pub(super) async fn handle_mcp_prompt_list_impl(&self) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let response = self
            .runtime
            .list_mcp_prompts()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_prompt_get_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpPromptGetParams = parse_params(params)?;
        let response = self
            .runtime
            .get_mcp_prompt(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_resource_list_impl(&self) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let response = self
            .runtime
            .list_mcp_resources()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_resource_subscribe_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpResourceSubscribeParams = parse_params(params)?;
        let response = self
            .runtime
            .subscribe_mcp_resource(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_resource_unsubscribe_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpResourceUnsubscribeParams = parse_params(params)?;
        let response = self
            .runtime
            .unsubscribe_mcp_resource(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }
}
