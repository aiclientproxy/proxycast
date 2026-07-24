use crate::active_time::active_time_timeout;
use crate::client_service::LimeMcpClientService;
use rmcp::handler::client::progress::ProgressSubscriber;
use rmcp::model::{
    CallToolRequest, CallToolRequestParam, CallToolResult, CancelledNotification,
    CancelledNotificationMethod, CancelledNotificationParam, ClientRequest, Extensions, JsonObject,
    ListToolsRequest, ListToolsResult, PaginatedRequestParam, ProgressToken, ServerResult,
};
use rmcp::service::{PeerRequestOptions, RequestHandle, RunningService, ServiceError};
use rmcp::RoleClient;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tool_runtime::mcp_connection::McpCallScope;

const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

pub struct McpBridgeCall {
    pub progress_token: ProgressToken,
    pub progress: ProgressSubscriber,
    pub response:
        Pin<Box<dyn Future<Output = Result<CallToolResult, ServiceError>> + Send + 'static>>,
}

#[derive(Clone)]
pub struct McpBridgeClient {
    service: Arc<RunningService<RoleClient, LimeMcpClientService>>,
    request_timeout: Duration,
    tool_timeout: Duration,
}

impl McpBridgeClient {
    pub fn new(
        service: Arc<RunningService<RoleClient, LimeMcpClientService>>,
        tool_timeout: Duration,
    ) -> Self {
        Self {
            service,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            tool_timeout,
        }
    }

    pub async fn list_tools(
        &self,
        cursor: Option<String>,
        extensions: Extensions,
        cancel_token: CancellationToken,
    ) -> Result<ListToolsResult, ServiceError> {
        let res = self
            .send_request(
                ClientRequest::ListToolsRequest(ListToolsRequest {
                    params: Some(PaginatedRequestParam { cursor }),
                    method: Default::default(),
                    extensions,
                }),
                cancel_token,
                self.request_timeout,
            )
            .await?;

        match res {
            ServerResult::ListToolsResult(result) => Ok(result),
            _ => Err(ServiceError::UnexpectedResponse),
        }
    }

    pub async fn call_tool(
        &self,
        name: &str,
        arguments: Option<JsonObject>,
        extensions: Extensions,
        scope: Option<&McpCallScope>,
        cancel_token: CancellationToken,
    ) -> Result<CallToolResult, ServiceError> {
        let _owner = self
            .service
            .service()
            .handler()
            .enter_elicitation_owner(scope.cloned())
            .await;
        let res = self
            .send_request(
                ClientRequest::CallToolRequest(CallToolRequest {
                    params: CallToolRequestParam {
                        name: name.to_string().into(),
                        arguments,
                    },
                    method: Default::default(),
                    extensions,
                }),
                cancel_token,
                self.tool_timeout,
            )
            .await?;

        match res {
            ServerResult::CallToolResult(result) => Ok(result),
            _ => Err(ServiceError::UnexpectedResponse),
        }
    }

    pub async fn start_tool_call(
        &self,
        name: &str,
        arguments: Option<JsonObject>,
        extensions: Extensions,
        scope: Option<&McpCallScope>,
        cancel_token: CancellationToken,
    ) -> Result<McpBridgeCall, ServiceError> {
        let owner = self
            .service
            .service()
            .handler()
            .enter_elicitation_owner(scope.cloned())
            .await;
        let handle = self
            .service
            .send_cancellable_request(
                ClientRequest::CallToolRequest(CallToolRequest {
                    params: CallToolRequestParam {
                        name: name.to_string().into(),
                        arguments,
                    },
                    method: Default::default(),
                    extensions,
                }),
                PeerRequestOptions::no_options(),
            )
            .await?;
        let progress_token = handle.progress_token.clone();
        let progress = self
            .service
            .service()
            .handler()
            .subscribe_progress(progress_token.clone())
            .await;
        let service = Arc::clone(&self.service);
        let timeout = self.tool_timeout;
        let response = Box::pin(async move {
            let _owner = owner;
            let result = await_request(service, handle, cancel_token, timeout).await?;
            match result {
                ServerResult::CallToolResult(result) => Ok(result),
                _ => Err(ServiceError::UnexpectedResponse),
            }
        });

        Ok(McpBridgeCall {
            progress_token,
            progress,
            response,
        })
    }

    async fn send_request(
        &self,
        request: ClientRequest,
        cancel_token: CancellationToken,
        timeout: Duration,
    ) -> Result<ServerResult, ServiceError> {
        let handle = self
            .service
            .send_cancellable_request(request, PeerRequestOptions::no_options())
            .await?;

        await_request(Arc::clone(&self.service), handle, cancel_token, timeout).await
    }
}

async fn await_request(
    service: Arc<RunningService<RoleClient, LimeMcpClientService>>,
    handle: RequestHandle<RoleClient>,
    cancel_token: CancellationToken,
    timeout: Duration,
) -> Result<ServerResult, ServiceError> {
    let request_id = handle.id;
    let peer = handle.peer.clone();
    let response = active_time_timeout(
        timeout,
        service
            .service()
            .handler()
            .elicitation_pause_state()
            .subscribe(),
        handle.rx,
    );
    tokio::pin!(response);

    tokio::select! {
        result = &mut response => match result {
            Ok(result) => result.map_err(|_error| ServiceError::TransportClosed)?,
            Err(()) => {
                let _ = peer.send_notification(
                    CancelledNotification {
                        params: CancelledNotificationParam {
                            request_id,
                            reason: Some("timed out".to_owned()),
                        },
                        method: CancelledNotificationMethod,
                        extensions: Default::default(),
                    }
                    .into(),
                ).await;
                Err(ServiceError::Timeout{timeout})
            }
        },
        _ = cancel_token.cancelled() => {
            let _ = peer.send_notification(
                CancelledNotification {
                    params: CancelledNotificationParam {
                        request_id,
                        reason: Some("operation cancelled".to_owned()),
                    },
                    method: CancelledNotificationMethod,
                    extensions: Default::default(),
                }
                .into(),
            ).await;
            Err(ServiceError::Cancelled { reason: None })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use rmcp::model::{Content, ProgressNotificationParam, ServerCapabilities, ServerInfo};
    use rmcp::service::{RequestContext, RoleServer, RunningServiceCancellationToken};
    use rmcp::{ServerHandler, ServiceExt};
    use tokio::sync::Notify;

    #[derive(Clone)]
    struct CancellationAwareToolServer {
        request_started: CancellationToken,
        cancellation_observed: CancellationToken,
    }

    impl ServerHandler for CancellationAwareToolServer {
        fn get_info(&self) -> ServerInfo {
            ServerInfo {
                capabilities: ServerCapabilities::builder().enable_tools().build(),
                ..Default::default()
            }
        }

        async fn call_tool(
            &self,
            _request: CallToolRequestParam,
            context: RequestContext<RoleServer>,
        ) -> Result<CallToolResult, rmcp::ErrorData> {
            self.request_started.cancel();
            context.ct.cancelled().await;
            self.cancellation_observed.cancel();
            Ok(CallToolResult::success(vec![rmcp::model::Content::text(
                "cancelled",
            )]))
        }
    }

    async fn start_cancellation_aware_client(
        tool_timeout: Duration,
    ) -> (
        McpBridgeClient,
        CancellationToken,
        CancellationToken,
        RunningServiceCancellationToken,
        tokio::task::JoinHandle<()>,
    ) {
        let request_started = CancellationToken::new();
        let cancellation_observed = CancellationToken::new();
        let (server_transport, client_transport) = tokio::io::duplex(4096);
        let server = CancellationAwareToolServer {
            request_started: request_started.clone(),
            cancellation_observed: cancellation_observed.clone(),
        };
        let server_task = tokio::spawn(async move {
            let service = server
                .serve(server_transport)
                .await
                .expect("start cancellation-aware MCP server");
            service
                .waiting()
                .await
                .expect("wait for cancellation-aware MCP server");
        });
        let service = LimeMcpClientService::new("timeout-server".to_string(), None)
            .serve(client_transport)
            .await
            .expect("start Lime MCP client");
        let service_cancellation = service.cancellation_token();
        let client = McpBridgeClient::new(Arc::new(service), tool_timeout);
        (
            client,
            request_started,
            cancellation_observed,
            service_cancellation,
            server_task,
        )
    }

    async fn stop_cancellation_aware_client(
        service_cancellation: RunningServiceCancellationToken,
        server_task: tokio::task::JoinHandle<()>,
    ) {
        service_cancellation.cancel();
        tokio::time::timeout(Duration::from_secs(2), server_task)
            .await
            .expect("cancellation-aware MCP server did not stop")
            .expect("cancellation-aware MCP server task failed");
    }

    #[tokio::test]
    async fn tool_timeout_notifies_server_request_cancellation() {
        let tool_timeout = Duration::from_millis(25);
        let (client, _request_started, cancellation_observed, service_cancellation, server_task) =
            start_cancellation_aware_client(tool_timeout).await;

        let result = client
            .call_tool(
                "wait-for-cancellation",
                None,
                Default::default(),
                None,
                CancellationToken::new(),
            )
            .await;

        assert!(
            matches!(
                result,
                Err(ServiceError::Timeout { timeout }) if timeout == tool_timeout
            ),
            "tool call must return its captured per-server timeout: {result:?}"
        );
        tokio::time::timeout(Duration::from_secs(2), cancellation_observed.cancelled())
            .await
            .expect("server request token was not cancelled after client timeout");

        stop_cancellation_aware_client(service_cancellation, server_task).await;
    }

    #[tokio::test]
    async fn caller_cancellation_notifies_in_flight_server_request() {
        let (client, request_started, cancellation_observed, service_cancellation, server_task) =
            start_cancellation_aware_client(Duration::from_secs(10)).await;
        let call_cancellation = CancellationToken::new();
        let cancellation = call_cancellation.clone();
        let call_task = tokio::spawn(async move {
            client
                .call_tool(
                    "wait-for-cancellation",
                    None,
                    Default::default(),
                    None,
                    cancellation,
                )
                .await
        });

        tokio::time::timeout(Duration::from_secs(2), request_started.cancelled())
            .await
            .expect("server did not receive the tool request before caller cancellation");
        call_cancellation.cancel();
        let result = tokio::time::timeout(Duration::from_secs(2), call_task)
            .await
            .expect("cancelled tool call did not return")
            .expect("cancelled tool call task failed");

        assert!(
            matches!(result, Err(ServiceError::Cancelled { reason: None })),
            "tool call must return caller cancellation: {result:?}"
        );
        tokio::time::timeout(Duration::from_secs(2), cancellation_observed.cancelled())
            .await
            .expect("server request token was not cancelled after caller cancellation");

        stop_cancellation_aware_client(service_cancellation, server_task).await;
    }

    #[derive(Clone)]
    struct ProgressToolServer {
        release_a: Arc<Notify>,
        release_b: Arc<Notify>,
    }

    impl ServerHandler for ProgressToolServer {
        fn get_info(&self) -> ServerInfo {
            ServerInfo {
                capabilities: ServerCapabilities::builder().enable_tools().build(),
                ..Default::default()
            }
        }

        async fn call_tool(
            &self,
            request: CallToolRequestParam,
            context: RequestContext<RoleServer>,
        ) -> Result<CallToolResult, rmcp::ErrorData> {
            let label = request
                .arguments
                .as_ref()
                .and_then(|arguments| arguments.get("label"))
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| rmcp::ErrorData::invalid_params("missing label", None))?;
            let release = match label {
                "A" => Arc::clone(&self.release_a),
                "B" => Arc::clone(&self.release_b),
                _ => return Err(rmcp::ErrorData::invalid_params("unknown label", None)),
            };
            release.notified().await;
            let progress_token = context
                .meta
                .get_progress_token()
                .ok_or_else(|| rmcp::ErrorData::invalid_params("missing progress token", None))?;
            context
                .peer
                .notify_progress(ProgressNotificationParam {
                    progress_token,
                    progress: 1.0,
                    total: Some(1.0),
                    message: Some(format!("progress-{label}")),
                })
                .await
                .map_err(|error| rmcp::ErrorData::internal_error(error.to_string(), None))?;
            Ok(CallToolResult::success(vec![Content::text(format!(
                "result-{label}"
            ))]))
        }
    }

    async fn run_progress_call(
        client: McpBridgeClient,
        label: &'static str,
        release: Arc<Notify>,
    ) -> (
        ProgressToken,
        ProgressNotificationParam,
        CallToolResult,
        ProgressSubscriber,
    ) {
        let mut arguments = JsonObject::new();
        arguments.insert("label".to_string(), serde_json::json!(label));
        let McpBridgeCall {
            progress_token,
            mut progress,
            response,
        } = client
            .start_tool_call(
                "progress",
                Some(arguments),
                Default::default(),
                None,
                CancellationToken::new(),
            )
            .await
            .expect("start progress tool call");
        release.notify_one();
        let (response, notification) = tokio::join!(response, progress.next());
        (
            progress_token,
            notification.expect("call-scoped progress notification"),
            response.expect("progress tool result"),
            progress,
        )
    }

    #[tokio::test]
    async fn concurrent_callers_receive_only_their_request_progress() {
        let release_a = Arc::new(Notify::new());
        let release_b = Arc::new(Notify::new());
        let server = ProgressToolServer {
            release_a: Arc::clone(&release_a),
            release_b: Arc::clone(&release_b),
        };
        let (server_transport, client_transport) = tokio::io::duplex(4096);
        let server_task = tokio::spawn(async move {
            let service = server
                .serve(server_transport)
                .await
                .expect("start progress MCP server");
            service
                .waiting()
                .await
                .expect("wait for progress MCP server");
        });
        let service = LimeMcpClientService::new("progress-server".to_string(), None)
            .serve(client_transport)
            .await
            .expect("start progress MCP client");
        let service_cancellation = service.cancellation_token();
        let client = McpBridgeClient::new(Arc::new(service), Duration::from_secs(5));

        let (mut call_a, mut call_b) = tokio::join!(
            run_progress_call(client.clone(), "A", release_a),
            run_progress_call(client, "B", release_b),
        );

        assert_ne!(call_a.0, call_b.0);
        assert_eq!(call_a.1.progress_token, call_a.0);
        assert_eq!(call_b.1.progress_token, call_b.0);
        assert_eq!(call_a.1.message.as_deref(), Some("progress-A"));
        assert_eq!(call_b.1.message.as_deref(), Some("progress-B"));
        assert_eq!(
            call_a.2.content[0].as_text().map(|text| text.text.as_str()),
            Some("result-A")
        );
        assert_eq!(
            call_b.2.content[0].as_text().map(|text| text.text.as_str()),
            Some("result-B")
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(20), call_a.3.next())
                .await
                .is_err(),
            "call A must not receive call B progress"
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(20), call_b.3.next())
                .await
                .is_err(),
            "call B must not receive call A progress"
        );

        service_cancellation.cancel();
        tokio::time::timeout(Duration::from_secs(2), server_task)
            .await
            .expect("progress MCP server did not stop")
            .expect("progress MCP server task failed");
    }
}
