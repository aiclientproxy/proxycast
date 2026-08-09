use super::stream::{ResponsesEventReducer, DEFAULT_STREAM_IDLE_TIMEOUT};
use super::{CurrentProviderError, CurrentProviderStream};
use async_stream::try_stream;
use futures::{SinkExt, StreamExt};
use serde_json::Value;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::net::TcpStream;
use tokio::sync::{Mutex, MutexGuard};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

pub(super) type ResponsesSocket = WebSocketStream<MaybeTlsStream<TcpStream>>;

struct SocketLease<'a> {
    connection: MutexGuard<'a, Option<ResponsesSocket>>,
    reusable: bool,
}

impl SocketLease<'_> {
    fn socket(&mut self) -> Result<&mut ResponsesSocket, CurrentProviderError> {
        self.connection.as_mut().ok_or_else(|| {
            CurrentProviderError::transport("Responses WebSocket connection is not initialized")
        })
    }

    fn keep(&mut self) {
        self.reusable = true;
    }
}

impl Drop for SocketLease<'_> {
    fn drop(&mut self) {
        if !self.reusable {
            self.connection.take();
        }
    }
}

pub(super) fn responses_websocket(
    connection: Arc<Mutex<Option<ResponsesSocket>>>,
    payload: Value,
    http_fallback: Arc<AtomicBool>,
    server_model: Option<String>,
    allow_openai_response_metadata: bool,
) -> CurrentProviderStream {
    Box::pin(try_stream! {
        let mut lease = SocketLease {
            connection: connection.lock().await,
            reusable: false,
        };
        let request = websocket_request(payload)?;
        lease.socket()?.send(Message::Text(request)).await.map_err(|error| {
            activate_fallback(
                &http_fallback,
                format!("发送 Responses WebSocket request 失败: {error}"),
            )
        })?;

        let mut reducer = ResponsesEventReducer::new(
            server_model.clone(),
            allow_openai_response_metadata,
        );
        if let Some(model) = server_model {
            yield runtime_core::CanonicalLlmEvent::ServerModel { model };
        }
        loop {
            let message = tokio::time::timeout(DEFAULT_STREAM_IDLE_TIMEOUT, lease.socket()?.next())
                .await
                .map_err(|_| {
                    activate_fallback(
                        &http_fallback,
                        "Responses WebSocket 等待 event 超时".to_string(),
                    )
                })?
                .ok_or_else(|| {
                    activate_fallback(
                        &http_fallback,
                        "Responses WebSocket 在 response.completed 前关闭".to_string(),
                    )
                })?
                .map_err(|error| {
                    activate_fallback(
                        &http_fallback,
                        format!("读取 Responses WebSocket event 失败: {error}"),
                    )
                })?;

            match message {
                Message::Text(text) => {
                    let payload: Value = serde_json::from_str(&text).map_err(|error| {
                        activate_fallback(
                            &http_fallback,
                            format!("解析 Responses WebSocket event 失败: {error}"),
                        )
                    })?;
                    if is_connection_limit_error(&payload) {
                        Err(activate_fallback(
                            &http_fallback,
                            "Responses WebSocket connection limit reached".to_string(),
                        ))?;
                    }
                    let batch = reducer.push(&payload).map_err(|error| {
                        http_fallback.store(true, Ordering::Release);
                        error
                    })?;
                    if batch.terminal {
                        lease.keep();
                    }
                    for event in batch.events {
                        yield event;
                    }
                    if batch.terminal {
                        return;
                    }
                }
                Message::Ping(payload) => {
                    lease.socket()?.send(Message::Pong(payload)).await.map_err(|error| {
                        activate_fallback(
                            &http_fallback,
                            format!("响应 Responses WebSocket ping 失败: {error}"),
                        )
                    })?;
                }
                Message::Pong(_) => {}
                Message::Close(_) => {
                    Err(activate_fallback(
                        &http_fallback,
                        "Responses WebSocket 在 response.completed 前收到 close".to_string(),
                    ))?;
                }
                Message::Binary(_) | Message::Frame(_) => {
                    Err(activate_fallback(
                        &http_fallback,
                        "Responses WebSocket 收到非文本 event".to_string(),
                    ))?;
                }
            }
        }
    })
}

fn is_connection_limit_error(payload: &Value) -> bool {
    payload
        .pointer("/error/code")
        .or_else(|| payload.get("code"))
        .and_then(Value::as_str)
        == Some("websocket_connection_limit_reached")
}

fn websocket_request(payload: Value) -> Result<String, CurrentProviderError> {
    let mut payload = payload.as_object().cloned().ok_or_else(|| {
        CurrentProviderError::invalid_request("Responses WebSocket payload 必须是 JSON object")
    })?;
    payload.remove("stream");
    payload.remove("background");
    payload.insert(
        "type".to_string(),
        Value::String("response.create".to_string()),
    );
    serde_json::to_string(&payload).map_err(|error| {
        CurrentProviderError::invalid_request(format!(
            "序列化 Responses WebSocket request 失败: {error}"
        ))
    })
}

fn activate_fallback(state: &AtomicBool, message: String) -> CurrentProviderError {
    state.store(true, Ordering::Release);
    CurrentProviderError::transport(message)
}

#[cfg(test)]
mod tests {
    use super::super::{
        CurrentProviderClient, CurrentProviderContent, CurrentProviderMessage,
        CurrentProviderRequest,
    };
    use super::websocket_request;
    use crate::runtime_provider::{RuntimeProviderConfig, RuntimeProviderProtocol};
    use futures::{SinkExt, StreamExt};
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    use tokio::net::TcpListener;
    use tokio::time::{timeout, Duration};
    use tokio_tungstenite::tungstenite::Message;

    #[test]
    fn websocket_request_wraps_response_create_and_removes_http_flags() {
        let request = websocket_request(json!({
            "model": "gpt-5.4",
            "input": [{ "role": "user", "content": "hello" }],
            "stream": true,
            "background": false
        }))
        .expect("websocket request");
        let request: serde_json::Value = serde_json::from_str(&request).expect("request json");

        assert_eq!(request["type"], "response.create");
        assert!(request.get("stream").is_none());
        assert!(request.get("background").is_none());
    }

    #[tokio::test]
    async fn client_reuses_one_websocket_for_sequential_response_requests() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind websocket reuse fixture");
        let address = listener
            .local_addr()
            .expect("websocket reuse fixture address");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let server_requests = Arc::clone(&requests);
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept websocket request");
            let mut socket = tokio_tungstenite::accept_async(stream)
                .await
                .expect("websocket handshake");
            for index in 1..=2 {
                let request = socket
                    .next()
                    .await
                    .expect("websocket request")
                    .expect("valid websocket request");
                let Message::Text(request) = request else {
                    panic!("expected text request");
                };
                server_requests
                    .lock()
                    .expect("record websocket request")
                    .push(
                        serde_json::from_str::<serde_json::Value>(&request).expect("request json"),
                    );
                socket
                    .send(Message::Text(
                        json!({
                            "type": "response.completed",
                            "response": {
                                "id": format!("resp-ws-{index}"),
                                "output": []
                            }
                        })
                        .to_string(),
                    ))
                    .await
                    .expect("send websocket response");
            }
        });
        let client = CurrentProviderClient::new(RuntimeProviderConfig {
            provider_name: "openai".to_string(),
            provider_selector: Some("openai".to_string()),
            model_name: "gpt-5.4".to_string(),
            api_key: Some("test-key".to_string()),
            auth: crate::runtime_provider::RuntimeProviderAuth::ApiKey,
            base_url: Some(format!("http://{address}")),
            api_version: None,
            credential_uuid: "credential-1".to_string(),
            reasoning_effort: None,
            service_tier: None,
            protocol: Some(RuntimeProviderProtocol::Responses),
            supports_websockets: true,
            toolshim: false,
            toolshim_model: None,
        })
        .expect("provider client");

        for _ in 0..2 {
            timeout(Duration::from_secs(2), async {
                client
                    .stream(CurrentProviderRequest::new(vec![
                        CurrentProviderMessage::user(vec![CurrentProviderContent::Text(
                            "hello".to_string(),
                        )]),
                    ]))
                    .await
                    .expect("websocket stream")
                    .collect::<Vec<_>>()
                    .await
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()
                    .expect("websocket events");
            })
            .await
            .expect("sequential websocket request timeout");
        }

        server.await.expect("websocket reuse server");
        let requests = requests.lock().expect("websocket requests");
        assert_eq!(requests.len(), 2);
        assert!(requests
            .iter()
            .all(|request| request["type"] == "response.create"));
    }
}
