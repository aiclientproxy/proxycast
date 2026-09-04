use crate::{AppServerClient, ClientError, ClientEvent};
use app_server_protocol::protocol::v2::{ServerNotification, ServerRequest};
use app_server_protocol::{
    InitializeParams, InitializeResponse, JsonRpcError, JsonRpcErrorResponse, JsonRpcMessage,
    JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, RequestId,
};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::io;
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};

use crate::transport::{SessionTransport, StdioTransport, StdioTransportConfig};

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("app-server session is closed")]
    Closed,
    #[error("app-server transport failed: {0}")]
    Transport(#[from] io::Error),
    #[error("failed to serialize {method} params: {source}")]
    Serialize {
        method: String,
        source: serde_json::Error,
    },
    #[error("failed to build {method} request: {source}")]
    BuildRequest { method: String, source: ClientError },
    #[error("{method} failed: {}", error.message)]
    Server { method: String, error: JsonRpcError },
    #[error("failed to decode {method} response: {source}")]
    Decode {
        method: String,
        source: serde_json::Error,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum SessionEvent {
    Notification(Box<ServerNotification>),
    RawNotification(JsonRpcNotification),
    ServerRequest(Box<ServerRequest>),
    RawServerRequest(JsonRpcRequest),
    Disconnected { message: String },
}

enum SessionCommand {
    Request {
        method: String,
        params: Value,
        response_tx: oneshot::Sender<Result<Value, SessionError>>,
    },
    Notify {
        notification: JsonRpcNotification,
        response_tx: oneshot::Sender<Result<(), SessionError>>,
    },
    Respond {
        message: JsonRpcMessage,
        response_tx: oneshot::Sender<Result<(), SessionError>>,
    },
    Shutdown {
        response_tx: oneshot::Sender<Result<(), SessionError>>,
    },
}

struct PendingRequest {
    method: String,
    response_tx: oneshot::Sender<Result<Value, SessionError>>,
}

#[derive(Clone)]
pub struct RequestHandle {
    command_tx: mpsc::Sender<SessionCommand>,
}

impl RequestHandle {
    pub async fn request<P, R>(
        &self,
        method: impl Into<String>,
        params: P,
    ) -> Result<R, SessionError>
    where
        P: Serialize,
        R: DeserializeOwned,
    {
        let method = method.into();
        let params = serde_json::to_value(params).map_err(|source| SessionError::Serialize {
            method: method.clone(),
            source,
        })?;
        let value = self.request_value(method.clone(), params).await?;
        serde_json::from_value(value).map_err(|source| SessionError::Decode { method, source })
    }

    pub async fn request_value(
        &self,
        method: impl Into<String>,
        params: Value,
    ) -> Result<Value, SessionError> {
        let (response_tx, response_rx) = oneshot::channel();
        self.command_tx
            .send(SessionCommand::Request {
                method: method.into(),
                params,
                response_tx,
            })
            .await
            .map_err(|_| SessionError::Closed)?;
        response_rx.await.map_err(|_| SessionError::Closed)?
    }

    pub async fn notify(&self, notification: JsonRpcNotification) -> Result<(), SessionError> {
        let (response_tx, response_rx) = oneshot::channel();
        self.command_tx
            .send(SessionCommand::Notify {
                notification,
                response_tx,
            })
            .await
            .map_err(|_| SessionError::Closed)?;
        response_rx.await.map_err(|_| SessionError::Closed)?
    }

    pub async fn respond<T: Serialize>(
        &self,
        id: RequestId,
        result: T,
    ) -> Result<(), SessionError> {
        let response =
            JsonRpcResponse::new(id, result).map_err(|source| SessionError::Serialize {
                method: "server request response".to_string(),
                source,
            })?;
        self.send_response(JsonRpcMessage::Response(response)).await
    }

    pub async fn reject(&self, id: RequestId, error: JsonRpcError) -> Result<(), SessionError> {
        self.send_response(JsonRpcMessage::Error(JsonRpcErrorResponse { id, error }))
            .await
    }

    async fn send_response(&self, message: JsonRpcMessage) -> Result<(), SessionError> {
        let (response_tx, response_rx) = oneshot::channel();
        self.command_tx
            .send(SessionCommand::Respond {
                message,
                response_tx,
            })
            .await
            .map_err(|_| SessionError::Closed)?;
        response_rx.await.map_err(|_| SessionError::Closed)?
    }
}

pub struct ClientSession {
    request_handle: RequestHandle,
    event_rx: mpsc::UnboundedReceiver<SessionEvent>,
    worker: tokio::task::JoinHandle<()>,
    initialize_response: InitializeResponse,
}

impl ClientSession {
    pub async fn start<T>(transport: T, initialize: InitializeParams) -> Result<Self, SessionError>
    where
        T: SessionTransport,
    {
        let (command_tx, command_rx) = mpsc::channel(64);
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let worker = tokio::spawn(run_session(transport, command_rx, event_tx));
        let request_handle = RequestHandle { command_tx };

        let initialize_response = match request_handle
            .request(app_server_protocol::METHOD_INITIALIZE, initialize)
            .await
        {
            Ok(response) => response,
            Err(error) => {
                stop_worker(&request_handle, worker).await;
                return Err(error);
            }
        };
        if let Err(error) = request_handle
            .notify(JsonRpcNotification::new(
                app_server_protocol::METHOD_INITIALIZED,
                Some(serde_json::json!({})),
            ))
            .await
        {
            stop_worker(&request_handle, worker).await;
            return Err(error);
        }

        Ok(Self {
            request_handle,
            event_rx,
            worker,
            initialize_response,
        })
    }

    pub async fn start_stdio(
        config: StdioTransportConfig,
        initialize: InitializeParams,
    ) -> Result<Self, SessionError> {
        Self::start(StdioTransport::spawn(config).await?, initialize).await
    }

    pub fn request_handle(&self) -> RequestHandle {
        self.request_handle.clone()
    }

    pub fn initialize_response(&self) -> &InitializeResponse {
        &self.initialize_response
    }

    pub async fn next_event(&mut self) -> Option<SessionEvent> {
        self.event_rx.recv().await
    }

    pub async fn shutdown(self) -> Result<(), SessionError> {
        let (response_tx, response_rx) = oneshot::channel();
        let send_result = self
            .request_handle
            .command_tx
            .send(SessionCommand::Shutdown { response_tx })
            .await;
        if send_result.is_err() {
            let _ = self.worker.await;
            return Err(SessionError::Closed);
        }
        let result = response_rx.await.map_err(|_| SessionError::Closed);
        let _ = self.worker.await;
        result?
    }
}

async fn stop_worker(request_handle: &RequestHandle, worker: tokio::task::JoinHandle<()>) {
    let (response_tx, response_rx) = oneshot::channel();
    if request_handle
        .command_tx
        .send(SessionCommand::Shutdown { response_tx })
        .await
        .is_ok()
    {
        let _ = response_rx.await;
    }
    let _ = worker.await;
}

async fn run_session<T: SessionTransport>(
    mut transport: T,
    mut command_rx: mpsc::Receiver<SessionCommand>,
    event_tx: mpsc::UnboundedSender<SessionEvent>,
) {
    let mut client = AppServerClient::new();
    let mut pending = HashMap::<RequestId, PendingRequest>::new();

    loop {
        tokio::select! {
            command = command_rx.recv() => {
                let Some(command) = command else {
                    let _ = transport.close().await;
                    fail_pending(&mut pending, "app-server command channel closed");
                    break;
                };
                match command {
                    SessionCommand::Request { method, params, response_tx } => {
                        let request = match client.request(method.clone(), params) {
                            Ok(request) => request,
                            Err(error) => {
                                let _ = response_tx.send(Err(SessionError::BuildRequest {
                                    method,
                                    source: error,
                                }));
                                continue;
                            }
                        };
                        let id = request.id.clone();
                        match transport.send(JsonRpcMessage::Request(request)).await {
                            Ok(()) => {
                                pending.insert(id, PendingRequest { method, response_tx });
                            }
                            Err(error) => {
                                let message = error.to_string();
                                let _ = response_tx.send(Err(SessionError::Transport(error)));
                                disconnect(&event_tx, &mut pending, message);
                                break;
                            }
                        }
                    }
                    SessionCommand::Notify { notification, response_tx } => {
                        let result = transport
                            .send(JsonRpcMessage::Notification(notification))
                            .await
                            .map_err(SessionError::Transport);
                        let failed = result.is_err();
                        let message = result.as_ref().err().map(ToString::to_string);
                        let _ = response_tx.send(result);
                        if failed {
                            disconnect(
                                &event_tx,
                                &mut pending,
                                message.unwrap_or_else(|| "app-server notification failed".to_string()),
                            );
                            break;
                        }
                    }
                    SessionCommand::Respond { message, response_tx } => {
                        let result = transport.send(message).await.map_err(SessionError::Transport);
                        let failed = result.is_err();
                        let message = result.as_ref().err().map(ToString::to_string);
                        let _ = response_tx.send(result);
                        if failed {
                            disconnect(
                                &event_tx,
                                &mut pending,
                                message.unwrap_or_else(|| "app-server response failed".to_string()),
                            );
                            break;
                        }
                    }
                    SessionCommand::Shutdown { response_tx } => {
                        let result = transport.close().await.map_err(SessionError::Transport);
                        let _ = response_tx.send(result);
                        fail_pending(&mut pending, "app-server session shut down");
                        break;
                    }
                }
            }
            incoming = transport.receive() => {
                match incoming {
                    Ok(Some(message)) => handle_message(message, &event_tx, &mut pending),
                    Ok(None) => {
                        disconnect(&event_tx, &mut pending, "app-server transport closed".to_string());
                        break;
                    }
                    Err(error) => {
                        disconnect(&event_tx, &mut pending, error.to_string());
                        break;
                    }
                }
            }
        }
    }
}

fn handle_message(
    message: JsonRpcMessage,
    event_tx: &mpsc::UnboundedSender<SessionEvent>,
    pending: &mut HashMap<RequestId, PendingRequest>,
) {
    match message {
        JsonRpcMessage::Response(response) => {
            if let Some(pending) = pending.remove(&response.id) {
                let _ = pending.response_tx.send(Ok(response.result));
            }
        }
        JsonRpcMessage::Error(response) => {
            if let Some(pending) = pending.remove(&response.id) {
                let _ = pending.response_tx.send(Err(SessionError::Server {
                    method: pending.method,
                    error: response.error,
                }));
            }
        }
        message => match AppServerClient::event(message) {
            Ok(ClientEvent::Lifecycle(notification)) => {
                let _ = event_tx.send(SessionEvent::Notification(notification));
            }
            Ok(
                ClientEvent::AgentSession(notification) | ClientEvent::Notification(notification),
            ) => {
                let _ = event_tx.send(SessionEvent::RawNotification(notification));
            }
            Ok(ClientEvent::Request(request)) => {
                let event = ServerRequest::try_from(request.clone())
                    .map(Box::new)
                    .map(SessionEvent::ServerRequest)
                    .unwrap_or(SessionEvent::RawServerRequest(request));
                let _ = event_tx.send(event);
            }
            Ok(ClientEvent::Response(_) | ClientEvent::Error(_)) => {}
            Err(error) => {
                let _ = event_tx.send(SessionEvent::Disconnected {
                    message: error.to_string(),
                });
            }
        },
    }
}

fn disconnect(
    event_tx: &mpsc::UnboundedSender<SessionEvent>,
    pending: &mut HashMap<RequestId, PendingRequest>,
    message: String,
) {
    fail_pending(pending, &message);
    let _ = event_tx.send(SessionEvent::Disconnected { message });
}

fn fail_pending(pending: &mut HashMap<RequestId, PendingRequest>, message: &str) {
    for (_, pending) in pending.drain() {
        let error = io::Error::new(io::ErrorKind::BrokenPipe, message.to_string());
        let _ = pending
            .response_tx
            .send(Err(SessionError::Transport(error)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use tokio::sync::mpsc;

    struct MemoryTransport {
        outbound: mpsc::UnboundedSender<JsonRpcMessage>,
        inbound: mpsc::UnboundedReceiver<JsonRpcMessage>,
        close_tx: Option<oneshot::Sender<()>>,
    }

    #[async_trait]
    impl SessionTransport for MemoryTransport {
        async fn send(&mut self, message: JsonRpcMessage) -> io::Result<()> {
            self.outbound
                .send(message)
                .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "test server closed"))
        }

        async fn receive(&mut self) -> io::Result<Option<JsonRpcMessage>> {
            Ok(self.inbound.recv().await)
        }

        async fn close(&mut self) -> io::Result<()> {
            if let Some(close_tx) = self.close_tx.take() {
                let _ = close_tx.send(());
            }
            Ok(())
        }
    }

    fn transport_pair() -> (
        MemoryTransport,
        mpsc::UnboundedReceiver<JsonRpcMessage>,
        mpsc::UnboundedSender<JsonRpcMessage>,
        oneshot::Receiver<()>,
    ) {
        let (client_tx, server_rx) = mpsc::unbounded_channel();
        let (server_tx, client_rx) = mpsc::unbounded_channel();
        let (close_tx, close_rx) = oneshot::channel();
        (
            MemoryTransport {
                outbound: client_tx,
                inbound: client_rx,
                close_tx: Some(close_tx),
            },
            server_rx,
            server_tx,
            close_rx,
        )
    }

    fn initialize_params() -> InitializeParams {
        InitializeParams {
            client_info: app_server_protocol::ClientInfo {
                name: "terminal-test".to_string(),
                title: Some("Terminal Test".to_string()),
                version: Some("1".to_string()),
            },
            capabilities: app_server_protocol::ClientCapabilities::default(),
        }
    }

    async fn complete_handshake(
        server_rx: &mut mpsc::UnboundedReceiver<JsonRpcMessage>,
        server_tx: &mpsc::UnboundedSender<JsonRpcMessage>,
    ) {
        let JsonRpcMessage::Request(initialize) = server_rx.recv().await.expect("initialize")
        else {
            panic!("expected initialize request");
        };
        assert_eq!(initialize.method, app_server_protocol::METHOD_INITIALIZE);
        server_tx
            .send(JsonRpcMessage::Response(
                JsonRpcResponse::new(
                    initialize.id,
                    app_server_protocol::InitializeResponse {
                        server_info: app_server_protocol::ServerInfo {
                            name: "test-server".to_string(),
                            version: "1".to_string(),
                            protocol_version: "2".to_string(),
                        },
                        platform: app_server_protocol::PlatformInfo {
                            family: "unix".to_string(),
                            os: "test".to_string(),
                        },
                        capabilities: app_server_protocol::ServerCapabilities {
                            agent_session: true,
                            capability_discovery: true,
                            artifact: false,
                            workspace: false,
                        },
                    },
                )
                .expect("response"),
            ))
            .expect("send initialize response");
        let JsonRpcMessage::Notification(initialized) =
            server_rx.recv().await.expect("initialized")
        else {
            panic!("expected initialized notification");
        };
        assert_eq!(initialized.method, app_server_protocol::METHOD_INITIALIZED);
    }

    #[tokio::test]
    async fn notification_does_not_block_request_response() {
        let (transport, mut server_rx, server_tx, _close_rx) = transport_pair();
        let server = tokio::spawn(async move {
            complete_handshake(&mut server_rx, &server_tx).await;
            let JsonRpcMessage::Request(request) = server_rx.recv().await.expect("request") else {
                panic!("expected request");
            };
            server_tx
                .send(JsonRpcMessage::Notification(JsonRpcNotification::new(
                    "terminal/test",
                    Some(serde_json::json!({ "sequence": 1 })),
                )))
                .expect("send notification");
            server_tx
                .send(JsonRpcMessage::Response(
                    JsonRpcResponse::new(request.id, serde_json::json!({ "value": 7 }))
                        .expect("response"),
                ))
                .expect("send response");
        });

        let mut session = ClientSession::start(transport, initialize_params())
            .await
            .expect("session");
        let response: Value = session
            .request_handle()
            .request("terminal/read", serde_json::json!({}))
            .await
            .expect("request response");

        assert_eq!(response, serde_json::json!({ "value": 7 }));
        assert!(matches!(
            session.next_event().await,
            Some(SessionEvent::RawNotification(notification))
                if notification.method == "terminal/test"
        ));
        server.await.expect("server task");
    }

    #[tokio::test]
    async fn reverse_request_is_typed_and_can_be_resolved() {
        let (transport, mut server_rx, server_tx, _close_rx) = transport_pair();
        let server = tokio::spawn(async move {
            complete_handshake(&mut server_rx, &server_tx).await;
            server_tx
                .send(JsonRpcMessage::Request(JsonRpcRequest::new(
                    RequestId::String("server-1".to_string()),
                    app_server_protocol::protocol::v2::METHOD_CURRENT_TIME_READ,
                    Some(serde_json::json!({ "threadId": "thread-1" })),
                )))
                .expect("send server request");
            let JsonRpcMessage::Response(response) = server_rx.recv().await.expect("response")
            else {
                panic!("expected response");
            };
            assert_eq!(response.id, RequestId::String("server-1".to_string()));
            assert_eq!(response.result, serde_json::json!({ "currentTimeAt": 42 }));
        });

        let mut session = ClientSession::start(transport, initialize_params())
            .await
            .expect("session");
        let request = session.next_event().await.expect("server request");
        let SessionEvent::ServerRequest(request) = request else {
            panic!("expected typed current time request");
        };
        let ServerRequest::CurrentTimeRead { id, params } = *request else {
            panic!("expected typed current time request");
        };
        assert_eq!(params.thread_id, "thread-1");
        session
            .request_handle()
            .respond(
                id,
                app_server_protocol::protocol::v2::CurrentTimeReadResponse {
                    current_time_at: 42,
                },
            )
            .await
            .expect("respond");
        server.await.expect("server task");
    }

    #[tokio::test]
    async fn server_error_keeps_method_context() {
        let (transport, mut server_rx, server_tx, _close_rx) = transport_pair();
        let server = tokio::spawn(async move {
            complete_handshake(&mut server_rx, &server_tx).await;
            let JsonRpcMessage::Request(request) = server_rx.recv().await.expect("request") else {
                panic!("expected request");
            };
            server_tx
                .send(JsonRpcMessage::Error(JsonRpcErrorResponse {
                    id: request.id,
                    error: JsonRpcError::new(-32000, "boom"),
                }))
                .expect("send error");
        });

        let session = ClientSession::start(transport, initialize_params())
            .await
            .expect("session");
        let error = session
            .request_handle()
            .request_value("turn/start", serde_json::json!({}))
            .await
            .expect_err("server error");

        assert!(matches!(
            error,
            SessionError::Server { method, error }
                if method == "turn/start" && error.message == "boom"
        ));
        server.await.expect("server task");
    }

    #[tokio::test]
    async fn shutdown_closes_transport() {
        let (transport, mut server_rx, server_tx, close_rx) = transport_pair();
        let server = tokio::spawn(async move {
            complete_handshake(&mut server_rx, &server_tx).await;
            assert!(server_rx.recv().await.is_none());
        });

        let session = ClientSession::start(transport, initialize_params())
            .await
            .expect("session");
        session.shutdown().await.expect("shutdown");

        close_rx.await.expect("transport close");
        server.await.expect("server task");
    }

    #[tokio::test]
    async fn initialize_failure_closes_transport() {
        let (transport, mut server_rx, server_tx, close_rx) = transport_pair();
        let server = tokio::spawn(async move {
            let JsonRpcMessage::Request(initialize) = server_rx.recv().await.expect("initialize")
            else {
                panic!("expected initialize request");
            };
            server_tx
                .send(JsonRpcMessage::Error(JsonRpcErrorResponse {
                    id: initialize.id,
                    error: JsonRpcError::new(-32000, "initialize failed"),
                }))
                .expect("send initialize error");
            assert!(server_rx.recv().await.is_none());
        });

        let error = match ClientSession::start(transport, initialize_params()).await {
            Ok(_) => panic!("initialize should fail"),
            Err(error) => error,
        };

        assert!(matches!(
            error,
            SessionError::Server { method, error }
                if method == app_server_protocol::METHOD_INITIALIZE
                    && error.message == "initialize failed"
        ));
        close_rx.await.expect("transport close");
        server.await.expect("server task");
    }

    #[tokio::test]
    async fn concurrent_requests_are_correlated_when_responses_arrive_out_of_order() {
        let (transport, mut server_rx, server_tx, _close_rx) = transport_pair();
        let server = tokio::spawn(async move {
            complete_handshake(&mut server_rx, &server_tx).await;
            let JsonRpcMessage::Request(first) = server_rx.recv().await.expect("first request")
            else {
                panic!("expected first request");
            };
            let JsonRpcMessage::Request(second) = server_rx.recv().await.expect("second request")
            else {
                panic!("expected second request");
            };
            server_tx
                .send(JsonRpcMessage::Response(
                    JsonRpcResponse::new(second.id, serde_json::json!({ "method": second.method }))
                        .expect("second response"),
                ))
                .expect("send second response");
            server_tx
                .send(JsonRpcMessage::Response(
                    JsonRpcResponse::new(first.id, serde_json::json!({ "method": first.method }))
                        .expect("first response"),
                ))
                .expect("send first response");
        });

        let session = ClientSession::start(transport, initialize_params())
            .await
            .expect("session");
        let first_handle = session.request_handle();
        let second_handle = session.request_handle();
        let first = tokio::spawn(async move {
            first_handle
                .request_value("test/first", serde_json::json!({}))
                .await
        });
        let second = tokio::spawn(async move {
            second_handle
                .request_value("test/second", serde_json::json!({}))
                .await
        });

        let first = first.await.expect("first task").expect("first response");
        let second = second.await.expect("second task").expect("second response");
        assert_eq!(first["method"], "test/first");
        assert_eq!(second["method"], "test/second");
        server.await.expect("server task");
    }
}
