//! Authenticated remote App Server transport.
//!
//! The remote transport intentionally implements the same [`SessionTransport`]
//! boundary as local stdio. Cloud clients can therefore add an authenticated
//! endpoint without introducing a second request router or runtime state owner.

use crate::transport::SessionTransport;
use app_server_protocol::JsonRpcMessage;
use async_trait::async_trait;
use futures::{SinkExt, StreamExt};
use std::fmt;
use std::io::{self, ErrorKind};
use std::time::Duration;
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::header::AUTHORIZATION;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async_with_config, MaybeTlsStream, WebSocketStream};
use url::Url;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_WEBSOCKET_MESSAGE_SIZE: usize = 128 << 20;

/// Configuration for a remote App Server WebSocket connection.
#[derive(Clone, PartialEq, Eq)]
pub struct RemoteTransportConfig {
    pub websocket_url: String,
    pub auth_token: Option<String>,
    expected_server_name: Option<String>,
}

impl fmt::Debug for RemoteTransportConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RemoteTransportConfig")
            .field("websocket_url", &redacted_remote_url(&self.websocket_url))
            .field(
                "auth_token",
                &self.auth_token.as_ref().map(|_| "<redacted>"),
            )
            .finish()
    }
}

impl RemoteTransportConfig {
    pub fn new(websocket_url: impl Into<String>) -> Self {
        Self {
            websocket_url: websocket_url.into(),
            auth_token: None,
            expected_server_name: Some(app_server_protocol::SERVER_NAME.to_string()),
        }
    }

    pub fn with_auth_token(mut self, auth_token: impl Into<String>) -> Self {
        self.auth_token = Some(auth_token.into());
        self
    }

    pub fn with_optional_auth_token(mut self, auth_token: Option<String>) -> Self {
        self.auth_token = auth_token;
        self
    }

    pub fn with_expected_server_name(mut self, server_name: impl Into<String>) -> Self {
        self.expected_server_name = Some(server_name.into());
        self
    }
}

/// WebSocket implementation of the App Server session transport.
pub struct RemoteTransport {
    writer: futures::stream::SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, Message>,
    reader: futures::stream::SplitStream<WebSocketStream<MaybeTlsStream<TcpStream>>>,
    expected_server_name: Option<String>,
}

impl RemoteTransport {
    pub async fn connect(config: RemoteTransportConfig) -> io::Result<Self> {
        let url = Url::parse(&config.websocket_url).map_err(|error| {
            io::Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "invalid remote App Server URL `{}`: {error}",
                    config.websocket_url
                ),
            )
        })?;
        if !matches!(url.scheme(), "ws" | "wss") {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "remote App Server URL must use `ws://` or `wss://`: {}",
                    config.websocket_url
                ),
            ));
        }
        if !url.username().is_empty() || url.password().is_some() || url.fragment().is_some() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "remote App Server URL must not contain userinfo or a fragment",
            ));
        }
        if remote_url_has_sensitive_query(&url) {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "remote App Server URL must not contain credential query parameters",
            ));
        }
        if config
            .auth_token
            .as_deref()
            .is_some_and(|token| token.trim().is_empty())
        {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "remote App Server auth token must not be empty",
            ));
        }
        if config.auth_token.is_some() && !websocket_url_supports_auth_token(&url) {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "remote auth tokens require `wss://` or loopback `ws://` URLs; got `{}`",
                    config.websocket_url
                ),
            ));
        }

        let mut request = url.as_str().into_client_request().map_err(|error| {
            io::Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "invalid remote App Server URL `{}`: {error}",
                    config.websocket_url
                ),
            )
        })?;
        if let Some(token) = config.auth_token.as_deref() {
            let value = HeaderValue::from_str(&format!("Bearer {token}")).map_err(|error| {
                io::Error::new(
                    ErrorKind::InvalidInput,
                    format!("invalid remote authorization header value: {error}"),
                )
            })?;
            request.headers_mut().insert(AUTHORIZATION, value);
        }

        let websocket_config = WebSocketConfig {
            max_frame_size: Some(MAX_WEBSOCKET_MESSAGE_SIZE),
            max_message_size: Some(MAX_WEBSOCKET_MESSAGE_SIZE),
            ..WebSocketConfig::default()
        };
        let (stream, _) = tokio::time::timeout(
            CONNECT_TIMEOUT,
            connect_async_with_config(request, Some(websocket_config), false),
        )
        .await
        .map_err(|_| {
            io::Error::new(
                ErrorKind::TimedOut,
                format!(
                    "timed out connecting to remote App Server at `{}`",
                    config.websocket_url
                ),
            )
        })?
        .map_err(|error| {
            io::Error::other(format!(
                "failed to connect to remote App Server at `{}`: {error}",
                config.websocket_url
            ))
        })?;

        let (writer, reader) = stream.split();
        Ok(Self {
            writer,
            reader,
            expected_server_name: config.expected_server_name,
        })
    }
}

pub(crate) fn websocket_url_supports_auth_token(url: &Url) -> bool {
    match (url.scheme(), url.host()) {
        ("wss", Some(_)) => true,
        ("ws", Some(url::Host::Domain(domain))) => domain.eq_ignore_ascii_case("localhost"),
        ("ws", Some(url::Host::Ipv4(address))) => address.is_loopback(),
        ("ws", Some(url::Host::Ipv6(address))) => address.is_loopback(),
        _ => false,
    }
}

fn remote_url_has_sensitive_query(url: &Url) -> bool {
    url.query_pairs().any(|(key, _)| {
        matches!(
            key.to_ascii_lowercase().as_str(),
            "access_token"
                | "api_key"
                | "apikey"
                | "auth_token"
                | "authorization"
                | "bearer"
                | "token"
        )
    })
}

fn redacted_remote_url(value: &str) -> String {
    let Ok(mut url) = Url::parse(value) else {
        return value.to_string();
    };
    if !url.username().is_empty() {
        let _ = url.set_username("<redacted>");
    }
    if url.password().is_some() {
        let _ = url.set_password(Some("<redacted>"));
    }
    if url.fragment().is_some() {
        url.set_fragment(Some("<redacted>"));
    }
    if remote_url_has_sensitive_query(&url) {
        url.set_query(Some("<redacted>"));
    }
    url.to_string()
}

#[async_trait]
impl SessionTransport for RemoteTransport {
    async fn send(&mut self, message: JsonRpcMessage) -> io::Result<()> {
        let line = app_server_transport::encode_message(&message)
            .map_err(|error| io::Error::new(ErrorKind::InvalidData, error))?;
        self.writer
            .send(Message::Text(line))
            .await
            .map_err(|error| io::Error::new(ErrorKind::BrokenPipe, error))
    }

    async fn receive(&mut self) -> io::Result<Option<JsonRpcMessage>> {
        loop {
            let Some(message) = self.reader.next().await else {
                return Ok(None);
            };
            let message = message.map_err(|error| io::Error::new(ErrorKind::InvalidData, error))?;
            match message {
                Message::Text(text) => {
                    let decoded = app_server_transport::decode_message(&text)
                        .map_err(|error| io::Error::new(ErrorKind::InvalidData, error))?;
                    return Ok(Some(decoded));
                }
                Message::Ping(payload) => {
                    self.writer
                        .send(Message::Pong(payload))
                        .await
                        .map_err(|error| io::Error::new(ErrorKind::BrokenPipe, error))?;
                }
                Message::Pong(_) => {}
                Message::Close(_) => return Ok(None),
                Message::Binary(_) | Message::Frame(_) => {
                    return Err(io::Error::new(
                        ErrorKind::InvalidData,
                        "remote App Server sent a non-text WebSocket frame",
                    ));
                }
            }
        }
    }

    async fn close(&mut self) -> io::Result<()> {
        self.writer
            .close()
            .await
            .map_err(|error| io::Error::new(ErrorKind::BrokenPipe, error))
    }

    fn validate_initialize_response(
        &self,
        response: &app_server_protocol::InitializeResponse,
    ) -> io::Result<()> {
        let Some(expected) = self.expected_server_name.as_deref() else {
            return Ok(());
        };
        if response.server_info.name == expected {
            return Ok(());
        }
        Err(io::Error::new(
            ErrorKind::PermissionDenied,
            format!(
                "remote App Server identity mismatch: expected `{expected}`, got `{}`",
                response.server_info.name
            ),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ClientSession;
    use app_server_protocol::{
        ClientCapabilities, ClientInfo, InitializeParams, JsonRpcMessage, JsonRpcNotification,
        JsonRpcResponse, PlatformInfo, ServerCapabilities, ServerInfo,
    };
    use futures::{SinkExt, StreamExt};
    use tokio::net::TcpListener;
    use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};
    use tokio_tungstenite::{accept_async, accept_hdr_async};

    #[test]
    fn auth_tokens_require_secure_or_loopback_endpoints() {
        assert!(websocket_url_supports_auth_token(
            &Url::parse("wss://cloud.example/rpc").expect("wss URL")
        ));
        assert!(websocket_url_supports_auth_token(
            &Url::parse("ws://127.0.0.1:4222").expect("loopback URL")
        ));
        assert!(!websocket_url_supports_auth_token(
            &Url::parse("ws://cloud.example/rpc").expect("insecure URL")
        ));
    }

    #[test]
    fn remote_config_debug_redacts_auth_token() {
        let config = RemoteTransportConfig::new("wss://cloud.example/rpc")
            .with_auth_token("super-secret-token");
        let debug = format!("{config:?}");
        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("super-secret-token"));

        let config = RemoteTransportConfig::new(
            "wss://cloud.example/rpc?tenant=public&access_token=query-secret#fragment-secret",
        );
        let debug = format!("{config:?}");
        assert!(!debug.contains("query-secret"));
        assert!(!debug.contains("fragment-secret"));
    }

    #[tokio::test]
    async fn remote_connection_rejects_embedded_credentials_fragments_and_empty_tokens() {
        for url in [
            "wss://user:password@cloud.example/rpc",
            "wss://cloud.example/rpc#fragment",
        ] {
            let error = match RemoteTransport::connect(RemoteTransportConfig::new(url)).await {
                Ok(_) => panic!("unsafe URL metadata must fail closed"),
                Err(error) => error,
            };
            assert!(error.to_string().contains("must not contain"));
        }

        let error = match RemoteTransport::connect(
            RemoteTransportConfig::new("wss://cloud.example/rpc").with_auth_token("   "),
        )
        .await
        {
            Ok(_) => panic!("empty auth token must fail closed"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("auth token must not be empty"));

        let error = match RemoteTransport::connect(RemoteTransportConfig::new(
            "wss://cloud.example/rpc?access_token=query-secret",
        ))
        .await
        {
            Ok(_) => panic!("credential query parameter must fail closed"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("credential query parameters"));
    }

    #[tokio::test]
    async fn authenticated_websocket_roundtrip_preserves_jsonrpc_messages() {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind test server");
        let address = listener.local_addr().expect("server address");
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept client");
            let mut websocket =
                accept_hdr_async(stream, |request: &Request, response: Response| {
                    assert_eq!(
                        request
                            .headers()
                            .get(AUTHORIZATION)
                            .and_then(|value| value.to_str().ok()),
                        Some("Bearer local-test-token")
                    );
                    Ok(response)
                })
                .await
                .expect("upgrade websocket");
            let Some(Ok(Message::Text(text))) = websocket.next().await else {
                panic!("expected JSON-RPC notification");
            };
            websocket
                .send(Message::Text(text))
                .await
                .expect("echo notification");
        });

        let mut transport = RemoteTransport::connect(
            RemoteTransportConfig::new(format!("ws://{address}/rpc"))
                .with_auth_token("local-test-token"),
        )
        .await
        .expect("connect remote transport");
        let message = JsonRpcMessage::Notification(JsonRpcNotification::new(
            "test/notification",
            Some(serde_json::json!({ "ok": true })),
        ));
        transport.send(message.clone()).await.expect("send message");
        assert_eq!(
            transport.receive().await.expect("receive message"),
            Some(message)
        );
        transport.close().await.expect("close transport");
        server.await.expect("server task");
    }

    #[tokio::test]
    async fn client_session_start_remote_completes_handshake_and_request() {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind test server");
        let address = listener.local_addr().expect("server address");
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept client");
            let mut websocket = accept_async(stream).await.expect("upgrade websocket");
            let Some(Ok(Message::Text(initialize))) = websocket.next().await else {
                panic!("expected initialize request");
            };
            let JsonRpcMessage::Request(initialize) =
                app_server_transport::decode_message(&initialize).expect("decode initialize")
            else {
                panic!("expected JSON-RPC initialize request");
            };
            websocket
                .send(Message::Text(
                    app_server_transport::encode_message(&JsonRpcMessage::Response(
                        JsonRpcResponse::new(
                            initialize.id,
                            app_server_protocol::InitializeResponse {
                                server_info: ServerInfo {
                                    name: app_server_protocol::SERVER_NAME.to_string(),
                                    version: "1".to_string(),
                                    protocol_version: app_server_protocol::PROTOCOL_VERSION
                                        .to_string(),
                                },
                                platform: PlatformInfo {
                                    family: "unix".to_string(),
                                    os: "test".to_string(),
                                },
                                capabilities: ServerCapabilities {
                                    agent_session: true,
                                    capability_discovery: true,
                                    artifact: false,
                                    workspace: false,
                                },
                            },
                        )
                        .expect("initialize response"),
                    ))
                    .expect("encode initialize response"),
                ))
                .await
                .expect("send initialize response");
            let Some(Ok(Message::Text(initialized))) = websocket.next().await else {
                panic!("expected initialized notification");
            };
            assert!(matches!(
                app_server_transport::decode_message(&initialized).expect("decode initialized"),
                JsonRpcMessage::Notification(notification)
                    if notification.method == app_server_protocol::METHOD_INITIALIZED
            ));
            let Some(Ok(Message::Text(request))) = websocket.next().await else {
                panic!("expected request");
            };
            let JsonRpcMessage::Request(request) =
                app_server_transport::decode_message(&request).expect("decode request")
            else {
                panic!("expected JSON-RPC request");
            };
            websocket
                .send(Message::Text(
                    app_server_transport::encode_message(&JsonRpcMessage::Response(
                        JsonRpcResponse::new(request.id, serde_json::json!({"ok": true}))
                            .expect("request response"),
                    ))
                    .expect("encode request response"),
                ))
                .await
                .expect("send request response");
            let _ = websocket.next().await;
        });

        let initialize = InitializeParams {
            client_info: ClientInfo {
                name: "remote-test-client".to_string(),
                title: Some("Remote Test".to_string()),
                version: Some("1".to_string()),
            },
            capabilities: ClientCapabilities::default(),
        };
        let session = ClientSession::start_remote(
            RemoteTransportConfig::new(format!("ws://{address}/rpc")),
            initialize,
        )
        .await
        .expect("remote session");
        let response: serde_json::Value = session
            .request_handle()
            .request("test/remote", serde_json::json!({}))
            .await
            .expect("remote response");
        assert_eq!(response, serde_json::json!({"ok": true}));
        session.shutdown().await.expect("remote shutdown");
        server.await.expect("server task");
    }

    #[tokio::test]
    async fn remote_identity_mismatch_closes_before_initialized() {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind test server");
        let address = listener.local_addr().expect("server address");
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept client");
            let mut websocket = accept_async(stream).await.expect("upgrade websocket");
            let Some(Ok(Message::Text(initialize))) = websocket.next().await else {
                panic!("expected initialize request");
            };
            let JsonRpcMessage::Request(initialize) =
                app_server_transport::decode_message(&initialize).expect("decode initialize")
            else {
                panic!("expected initialize request");
            };
            let response = app_server_protocol::InitializeResponse {
                server_info: ServerInfo {
                    name: "unexpected-server".to_string(),
                    version: "1".to_string(),
                    protocol_version: app_server_protocol::PROTOCOL_VERSION.to_string(),
                },
                platform: PlatformInfo {
                    family: "unix".to_string(),
                    os: "test".to_string(),
                },
                capabilities: ServerCapabilities {
                    agent_session: true,
                    capability_discovery: true,
                    artifact: false,
                    workspace: false,
                },
            };
            websocket
                .send(Message::Text(
                    app_server_transport::encode_message(&JsonRpcMessage::Response(
                        JsonRpcResponse::new(initialize.id, response).expect("response"),
                    ))
                    .expect("encode initialize response"),
                ))
                .await
                .expect("send initialize response");
            match tokio::time::timeout(Duration::from_secs(2), websocket.next()).await {
                Ok(Some(Ok(Message::Text(text)))) => {
                    panic!("initialized must not be sent: {text}");
                }
                Ok(Some(Ok(Message::Close(_)))) | Ok(None) => {}
                Ok(Some(Ok(message))) => panic!("unexpected post-handshake message: {message:?}"),
                Ok(Some(Err(error))) => panic!("websocket error: {error}"),
                Err(_) => panic!("timed out waiting for client close"),
            }
        });

        let initialize = InitializeParams {
            client_info: ClientInfo {
                name: "remote-test-client".to_string(),
                title: Some("Remote Test".to_string()),
                version: Some("1".to_string()),
            },
            capabilities: ClientCapabilities::default(),
        };
        let error = match ClientSession::start_remote(
            RemoteTransportConfig::new(format!("ws://{address}/rpc")),
            initialize,
        )
        .await
        {
            Ok(_) => panic!("identity mismatch must fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("identity mismatch"));
        server.await.expect("server task");
    }

    #[tokio::test]
    async fn auth_tokens_fail_closed_for_non_loopback_ws() {
        let error = match RemoteTransport::connect(
            RemoteTransportConfig::new("ws://cloud.example/rpc").with_auth_token("secret"),
        )
        .await
        {
            Ok(_) => panic!("insecure remote auth must fail closed"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("wss://"));
    }
}
