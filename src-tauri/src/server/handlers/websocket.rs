//! WebSocket 连接处理器
//!
//! 处理 WebSocket 连接的建立、消息收发和 API 请求转发

use axum::{
    body::Body,
    extract::{
        ws::{Message as WsMessage, WebSocket, WebSocketUpgrade},
        Query, State,
    },
    http::HeaderMap,
    response::IntoResponse,
};
use futures::{SinkExt, StreamExt as FuturesStreamExt};
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::converter::anthropic_to_openai::convert_anthropic_to_openai;
use crate::converter::openai_to_antigravity::{
    convert_antigravity_to_openai_response, convert_openai_to_antigravity_with_context,
};
use crate::models::anthropic::AnthropicMessagesRequest;
use crate::models::openai::ChatCompletionRequest;
use crate::models::provider_pool_model::ProviderCredential;
use crate::processor::RequestContext;
use crate::providers::{
    AntigravityProvider, ClaudeCustomProvider, KiroProvider, OpenAICustomProvider,
};
use crate::server::AppState;
use crate::server_utils::parse_cw_response;
use crate::websocket::{
    WsApiRequest, WsApiResponse, WsEndpoint, WsError, WsFlowEvent, WsMessage as WsProtoMessage,
};

/// WebSocket 查询参数
#[derive(Debug, Deserialize, Default)]
pub struct WsQueryParams {
    /// API 密钥（通过 URL 参数传递）
    pub api_key: Option<String>,
    /// Token（通过 URL 参数传递，与 api_key 等效）
    pub token: Option<String>,
}

/// WebSocket 升级处理器
pub async fn ws_upgrade_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Query(params): Query<WsQueryParams>,
    headers: HeaderMap,
) -> impl IntoResponse {
    // 验证 API 密钥：优先从 header 获取，其次从 URL 参数获取
    let auth = headers
        .get("authorization")
        .or_else(|| headers.get("x-api-key"))
        .and_then(|v| v.to_str().ok());

    let key = match auth {
        Some(s) if s.starts_with("Bearer ") => Some(&s[7..]),
        Some(s) => Some(s),
        None => {
            // 尝试从 URL 参数获取
            params.api_key.as_deref().or(params.token.as_deref())
        }
    };

    // 如果没有提供任何认证信息，允许连接（用于内部 Flow Monitor）
    // 但会在日志中记录
    let authenticated = match key {
        Some(k) if k == state.api_key => true,
        Some(_) => {
            return axum::http::Response::builder()
                .status(401)
                .body(Body::from("Invalid API key"))
                .unwrap()
                .into_response();
        }
        None => {
            // 允许无认证连接（仅用于本地 Flow Monitor UI）
            tracing::debug!("[WS] Allowing unauthenticated connection for Flow Monitor");
            false
        }
    };

    // 获取客户端信息
    let client_info = headers
        .get("user-agent")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    ws.on_upgrade(move |socket| handle_websocket(socket, state, client_info, authenticated))
}

/// 处理 WebSocket 连接
pub async fn handle_websocket(
    socket: WebSocket,
    state: AppState,
    client_info: Option<String>,
    authenticated: bool,
) {
    let conn_id = uuid::Uuid::new_v4().to_string();

    // 注册连接
    if let Err(e) = state
        .ws_manager
        .register(conn_id.clone(), client_info.clone())
    {
        state.logs.write().await.add(
            "error",
            &format!("[WS] Failed to register connection: {}", e.message),
        );
        return;
    }

    state.logs.write().await.add(
        "info",
        &format!(
            "[WS] New connection: {} (client: {:?}, authenticated: {})",
            &conn_id[..8],
            client_info,
            authenticated
        ),
    );

    let (sender, mut receiver) = socket.split();
    let sender = Arc::new(Mutex::new(sender));

    // Flow 事件订阅状态
    let flow_subscribed = Arc::new(std::sync::atomic::AtomicBool::new(false));

    // 启动 Flow 事件转发任务
    let flow_sender = sender.clone();
    let flow_subscribed_clone = flow_subscribed.clone();
    let flow_monitor = state.flow_monitor.clone();
    let conn_id_clone = conn_id.clone();
    let _logs_clone = state.logs.clone();

    let flow_task = tokio::spawn(async move {
        let mut flow_receiver = flow_monitor.subscribe();

        loop {
            match flow_receiver.recv().await {
                Ok(event) => {
                    // 只有在订阅状态下才转发事件
                    if !flow_subscribed_clone.load(std::sync::atomic::Ordering::Relaxed) {
                        continue;
                    }

                    // 转换为 WebSocket 消息
                    let ws_event: WsFlowEvent = event.into();
                    let ws_msg = WsProtoMessage::FlowEvent(ws_event);

                    if let Ok(msg_text) = serde_json::to_string(&ws_msg) {
                        let mut sender_guard = flow_sender.lock().await;
                        if sender_guard
                            .send(WsMessage::Text(msg_text.into()))
                            .await
                            .is_err()
                        {
                            tracing::debug!(
                                "[WS] Flow event send failed for connection {}",
                                &conn_id_clone[..8]
                            );
                            break;
                        }
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!(
                        "[WS] Flow event receiver lagged by {} messages for connection {}",
                        n,
                        &conn_id_clone[..8]
                    );
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    tracing::debug!(
                        "[WS] Flow event channel closed for connection {}",
                        &conn_id_clone[..8]
                    );
                    break;
                }
            }
        }
    });

    // 消息处理循环
    while let Some(msg) = receiver.next().await {
        match msg {
            Ok(WsMessage::Text(text)) => {
                state.ws_manager.on_message();
                state.ws_manager.increment_request_count(&conn_id);

                match serde_json::from_str::<WsProtoMessage>(&text) {
                    Ok(ws_msg) => {
                        let response =
                            handle_ws_message(&state, &conn_id, ws_msg, &flow_subscribed).await;
                        if let Some(resp) = response {
                            let resp_text = serde_json::to_string(&resp).unwrap_or_default();
                            let mut sender_guard = sender.lock().await;
                            if sender_guard
                                .send(WsMessage::Text(resp_text.into()))
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        state.ws_manager.on_error();
                        let error = WsProtoMessage::Error(WsError::invalid_message(format!(
                            "Failed to parse message: {}",
                            e
                        )));
                        let error_text = serde_json::to_string(&error).unwrap_or_default();
                        let mut sender_guard = sender.lock().await;
                        if sender_guard
                            .send(WsMessage::Text(error_text.into()))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                }
            }
            Ok(WsMessage::Binary(_)) => {
                state.ws_manager.on_error();
                let error = WsProtoMessage::Error(WsError::invalid_message(
                    "Binary messages not supported",
                ));
                let error_text = serde_json::to_string(&error).unwrap_or_default();
                let mut sender_guard = sender.lock().await;
                if sender_guard
                    .send(WsMessage::Text(error_text.into()))
                    .await
                    .is_err()
                {
                    break;
                }
            }
            Ok(WsMessage::Ping(data)) => {
                let mut sender_guard = sender.lock().await;
                if sender_guard.send(WsMessage::Pong(data)).await.is_err() {
                    break;
                }
            }
            Ok(WsMessage::Pong(_)) => {
                // 收到 pong，连接正常
            }
            Ok(WsMessage::Close(_)) => {
                break;
            }
            Err(e) => {
                state.logs.write().await.add(
                    "error",
                    &format!("[WS] Connection {} error: {}", &conn_id[..8], e),
                );
                break;
            }
        }
    }

    // 取消 Flow 事件转发任务
    flow_task.abort();

    // 清理连接
    state.ws_manager.unregister(&conn_id);
    state.logs.write().await.add(
        "info",
        &format!("[WS] Connection closed: {}", &conn_id[..8]),
    );
}

/// 处理 WebSocket 消息
async fn handle_ws_message(
    state: &AppState,
    conn_id: &str,
    msg: WsProtoMessage,
    flow_subscribed: &Arc<std::sync::atomic::AtomicBool>,
) -> Option<WsProtoMessage> {
    match msg {
        WsProtoMessage::Ping { timestamp } => Some(WsProtoMessage::Pong { timestamp }),
        WsProtoMessage::Pong { .. } => None,
        WsProtoMessage::SubscribeFlowEvents => {
            // 订阅 Flow 事件
            flow_subscribed.store(true, std::sync::atomic::Ordering::Relaxed);
            state.logs.write().await.add(
                "info",
                &format!(
                    "[WS] Connection {} subscribed to flow events",
                    &conn_id[..8]
                ),
            );
            // 返回确认消息
            Some(WsProtoMessage::Response(WsApiResponse {
                request_id: "subscribe_flow_events".to_string(),
                payload: serde_json::json!({
                    "status": "subscribed",
                    "message": "Successfully subscribed to flow events"
                }),
            }))
        }
        WsProtoMessage::UnsubscribeFlowEvents => {
            // 取消订阅 Flow 事件
            flow_subscribed.store(false, std::sync::atomic::Ordering::Relaxed);
            state.logs.write().await.add(
                "info",
                &format!(
                    "[WS] Connection {} unsubscribed from flow events",
                    &conn_id[..8]
                ),
            );
            // 返回确认消息
            Some(WsProtoMessage::Response(WsApiResponse {
                request_id: "unsubscribe_flow_events".to_string(),
                payload: serde_json::json!({
                    "status": "unsubscribed",
                    "message": "Successfully unsubscribed from flow events"
                }),
            }))
        }
        WsProtoMessage::FlowEvent(_) => {
            // 客户端不应该发送 FlowEvent 消息
            Some(WsProtoMessage::Error(WsError::invalid_request(
                None,
                "FlowEvent messages are server-to-client only",
            )))
        }
        WsProtoMessage::Request(request) => {
            state.logs.write().await.add(
                "info",
                &format!(
                    "[WS] Request from {}: id={} endpoint={:?}",
                    &conn_id[..8],
                    request.request_id,
                    request.endpoint
                ),
            );

            // 处理 API 请求
            let response = handle_ws_api_request(state, &request).await;
            Some(response)
        }
        WsProtoMessage::Response(_)
        | WsProtoMessage::StreamChunk(_)
        | WsProtoMessage::StreamEnd(_) => Some(WsProtoMessage::Error(WsError::invalid_request(
            None,
            "Invalid message type from client",
        ))),
        WsProtoMessage::Error(_) => None,
        WsProtoMessage::SubscribeKiroEvents => {
            // TODO: 实现Kiro事件订阅
            Some(WsProtoMessage::Response(WsApiResponse {
                request_id: "subscribe_kiro_events".to_string(),
                payload: serde_json::json!({
                    "status": "subscribed",
                    "message": "Successfully subscribed to kiro events"
                }),
            }))
        }
        WsProtoMessage::UnsubscribeKiroEvents => {
            // TODO: 实现Kiro事件取消订阅
            Some(WsProtoMessage::Response(WsApiResponse {
                request_id: "unsubscribe_kiro_events".to_string(),
                payload: serde_json::json!({
                    "status": "unsubscribed",
                    "message": "Successfully unsubscribed from kiro events"
                }),
            }))
        }
        WsProtoMessage::KiroCredentialEvent(_) => {
            // Kiro事件是服务端到客户端的消息，客户端不应该发送
            Some(WsProtoMessage::Error(WsError::invalid_message(
                "KiroCredentialEvent messages are server-to-client only",
            )))
        }
    }
}

/// 处理 WebSocket API 请求
async fn handle_ws_api_request(state: &AppState, request: &WsApiRequest) -> WsProtoMessage {
    match request.endpoint {
        WsEndpoint::Models => {
            // 返回模型列表
            let models = serde_json::json!({
                "object": "list",
                "data": [
                    {"id": "claude-sonnet-4-5", "object": "model", "owned_by": "anthropic"},
                    {"id": "claude-sonnet-4-5-20250929", "object": "model", "owned_by": "anthropic"},
                    {"id": "claude-3-7-sonnet-20250219", "object": "model", "owned_by": "anthropic"},
                    {"id": "gemini-2.5-flash", "object": "model", "owned_by": "google"},
                    {"id": "gemini-2.5-pro", "object": "model", "owned_by": "google"},
                    {"id": "qwen3-coder-plus", "object": "model", "owned_by": "alibaba"},
                ]
            });
            WsProtoMessage::Response(WsApiResponse {
                request_id: request.request_id.clone(),
                payload: models,
            })
        }
        WsEndpoint::ChatCompletions => {
            // 解析 ChatCompletionRequest
            match serde_json::from_value::<ChatCompletionRequest>(request.payload.clone()) {
                Ok(chat_request) => {
                    handle_ws_chat_completions(state, &request.request_id, chat_request).await
                }
                Err(e) => WsProtoMessage::Error(WsError::invalid_request(
                    Some(request.request_id.clone()),
                    format!("Invalid chat completion request: {}", e),
                )),
            }
        }
        WsEndpoint::Messages => {
            // 解析 AnthropicMessagesRequest
            match serde_json::from_value::<AnthropicMessagesRequest>(request.payload.clone()) {
                Ok(messages_request) => {
                    handle_ws_anthropic_messages(state, &request.request_id, messages_request).await
                }
                Err(e) => WsProtoMessage::Error(WsError::invalid_request(
                    Some(request.request_id.clone()),
                    format!("Invalid messages request: {}", e),
                )),
            }
        }
    }
}

/// 处理 WebSocket chat completions 请求
async fn handle_ws_chat_completions(
    state: &AppState,
    request_id: &str,
    mut request: ChatCompletionRequest,
) -> WsProtoMessage {
    // 创建请求上下文
    let mut ctx = RequestContext::new(request.model.clone()).with_stream(request.stream);

    // 使用 RequestProcessor 解析模型别名和路由
    let _provider = state.processor.resolve_and_route(&mut ctx).await;

    // 更新请求中的模型名为解析后的模型
    if ctx.resolved_model != ctx.original_model {
        request.model = ctx.resolved_model.clone();
    }

    // 应用参数注入
    let injection_enabled = *state.injection_enabled.read().await;
    if injection_enabled {
        let injector = state.processor.injector.read().await;
        let mut payload = serde_json::to_value(&request).unwrap_or_default();
        let result = injector.inject(&request.model, &mut payload);
        if result.has_injections() {
            if let Ok(updated) = serde_json::from_value(payload) {
                request = updated;
            }
        }
    }

    // 获取默认 provider
    let default_provider = state.default_provider.read().await.clone();

    // 尝试从凭证池中选择凭证
    let credential = match &state.db {
        Some(db) => state
            .pool_service
            .select_credential(db, &default_provider, Some(&request.model))
            .ok()
            .flatten(),
        None => None,
    };

    // 如果找到凭证，使用它调用 API
    if let Some(cred) = credential {
        // 简化实现：直接调用 provider 并返回结果
        // 实际实现应该复用 call_provider_openai 的逻辑
        match call_provider_openai_for_ws(state, &cred, &request).await {
            Ok(response) => WsProtoMessage::Response(WsApiResponse {
                request_id: request_id.to_string(),
                payload: response,
            }),
            Err(e) => WsProtoMessage::Error(WsError::upstream(Some(request_id.to_string()), e)),
        }
    } else {
        // 回退到 Kiro provider
        let kiro = state.kiro.read().await;
        match kiro.call_api(&request).await {
            Ok(resp) => {
                if resp.status().is_success() {
                    match resp.text().await {
                        Ok(body) => {
                            let parsed = parse_cw_response(&body);
                            let has_tool_calls = !parsed.tool_calls.is_empty();

                            let message = if has_tool_calls {
                                serde_json::json!({
                                    "role": "assistant",
                                    "content": if parsed.content.is_empty() { serde_json::Value::Null } else { serde_json::json!(parsed.content) },
                                    "tool_calls": parsed.tool_calls.iter().map(|tc| {
                                        serde_json::json!({
                                            "id": tc.id,
                                            "type": "function",
                                            "function": {
                                                "name": tc.function.name,
                                                "arguments": tc.function.arguments
                                            }
                                        })
                                    }).collect::<Vec<_>>()
                                })
                            } else {
                                serde_json::json!({
                                    "role": "assistant",
                                    "content": parsed.content
                                })
                            };

                            let response = serde_json::json!({
                                "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                                "object": "chat.completion",
                                "created": std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_secs(),
                                "model": request.model,
                                "choices": [{
                                    "index": 0,
                                    "message": message,
                                    "finish_reason": if has_tool_calls { "tool_calls" } else { "stop" }
                                }],
                                "usage": {
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0,
                                    "total_tokens": 0
                                }
                            });

                            WsProtoMessage::Response(WsApiResponse {
                                request_id: request_id.to_string(),
                                payload: response,
                            })
                        }
                        Err(e) => WsProtoMessage::Error(WsError::internal(
                            Some(request_id.to_string()),
                            e.to_string(),
                        )),
                    }
                } else {
                    let body = resp.text().await.unwrap_or_default();
                    WsProtoMessage::Error(WsError::upstream(
                        Some(request_id.to_string()),
                        format!("Upstream error: {}", body),
                    ))
                }
            }
            Err(e) => WsProtoMessage::Error(WsError::internal(
                Some(request_id.to_string()),
                e.to_string(),
            )),
        }
    }
}

/// 处理 WebSocket anthropic messages 请求
async fn handle_ws_anthropic_messages(
    state: &AppState,
    request_id: &str,
    mut request: AnthropicMessagesRequest,
) -> WsProtoMessage {
    // 创建请求上下文
    let mut ctx = RequestContext::new(request.model.clone()).with_stream(request.stream);

    // 使用 RequestProcessor 解析模型别名和路由
    let _provider = state.processor.resolve_and_route(&mut ctx).await;

    // 更新请求中的模型名为解析后的模型
    if ctx.resolved_model != ctx.original_model {
        request.model = ctx.resolved_model.clone();
    }

    // 应用参数注入
    let injection_enabled = *state.injection_enabled.read().await;
    if injection_enabled {
        let injector = state.processor.injector.read().await;
        let mut payload = serde_json::to_value(&request).unwrap_or_default();
        let result = injector.inject(&request.model, &mut payload);
        if result.has_injections() {
            if let Ok(updated) = serde_json::from_value(payload) {
                request = updated;
            }
        }
    }

    // 获取默认 provider
    let default_provider = state.default_provider.read().await.clone();

    // 尝试从凭证池中选择凭证
    let credential = match &state.db {
        Some(db) => state
            .pool_service
            .select_credential(db, &default_provider, Some(&request.model))
            .ok()
            .flatten(),
        None => None,
    };

    // 如果找到凭证，使用它调用 API
    if let Some(cred) = credential {
        match call_provider_anthropic_for_ws(state, &cred, &request).await {
            Ok(response) => WsProtoMessage::Response(WsApiResponse {
                request_id: request_id.to_string(),
                payload: response,
            }),
            Err(e) => WsProtoMessage::Error(WsError::upstream(Some(request_id.to_string()), e)),
        }
    } else {
        // 回退到 Kiro provider
        let kiro = state.kiro.read().await;

        // 转换为 OpenAI 格式
        let openai_request = convert_anthropic_to_openai(&request);

        match kiro.call_api(&openai_request).await {
            Ok(resp) => {
                if resp.status().is_success() {
                    match resp.text().await {
                        Ok(body) => {
                            let parsed = parse_cw_response(&body);

                            // 转换为 Anthropic 格式响应
                            let response = serde_json::json!({
                                "id": format!("msg_{}", uuid::Uuid::new_v4()),
                                "type": "message",
                                "role": "assistant",
                                "content": [{
                                    "type": "text",
                                    "text": parsed.content
                                }],
                                "model": request.model,
                                "stop_reason": "end_turn",
                                "usage": {
                                    "input_tokens": 0,
                                    "output_tokens": 0
                                }
                            });

                            WsProtoMessage::Response(WsApiResponse {
                                request_id: request_id.to_string(),
                                payload: response,
                            })
                        }
                        Err(e) => WsProtoMessage::Error(WsError::internal(
                            Some(request_id.to_string()),
                            e.to_string(),
                        )),
                    }
                } else {
                    let body = resp.text().await.unwrap_or_default();
                    WsProtoMessage::Error(WsError::upstream(
                        Some(request_id.to_string()),
                        format!("Upstream error: {}", body),
                    ))
                }
            }
            Err(e) => WsProtoMessage::Error(WsError::internal(
                Some(request_id.to_string()),
                e.to_string(),
            )),
        }
    }
}

/// WebSocket 专用的 OpenAI 格式 Provider 调用
pub async fn call_provider_openai_for_ws(
    state: &AppState,
    credential: &ProviderCredential,
    request: &ChatCompletionRequest,
) -> Result<serde_json::Value, String> {
    use crate::models::provider_pool_model::CredentialData;

    match &credential.credential {
        CredentialData::KiroOAuth { creds_file_path } => {
            let mut kiro = KiroProvider::new();
            if let Err(e) = kiro.load_credentials_from_path(creds_file_path).await {
                if let Some(db) = &state.db {
                    let _ = state.pool_service.mark_unhealthy(
                        db,
                        &credential.uuid,
                        Some(&format!("Failed to load credentials: {}", e)),
                    );
                }
                return Err(e.to_string());
            }
            if let Err(e) = kiro.refresh_token().await {
                if let Some(db) = &state.db {
                    let _ = state.pool_service.mark_unhealthy(
                        db,
                        &credential.uuid,
                        Some(&format!("Token refresh failed: {}", e)),
                    );
                }
                return Err(e.to_string());
            }

            let resp = match kiro.call_api(request).await {
                Ok(r) => r,
                Err(e) => {
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                    }
                    return Err(e.to_string());
                }
            };
            if resp.status().is_success() {
                let body = resp.text().await.map_err(|e| e.to_string())?;
                let parsed = parse_cw_response(&body);
                let has_tool_calls = !parsed.tool_calls.is_empty();

                // 记录成功
                if let Some(db) = &state.db {
                    let _ =
                        state
                            .pool_service
                            .mark_healthy(db, &credential.uuid, Some(&request.model));
                    let _ = state.pool_service.record_usage(db, &credential.uuid);
                }

                let message = if has_tool_calls {
                    serde_json::json!({
                        "role": "assistant",
                        "content": if parsed.content.is_empty() { serde_json::Value::Null } else { serde_json::json!(parsed.content) },
                        "tool_calls": parsed.tool_calls.iter().map(|tc| {
                            serde_json::json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            })
                        }).collect::<Vec<_>>()
                    })
                } else {
                    serde_json::json!({
                        "role": "assistant",
                        "content": parsed.content
                    })
                };

                Ok(serde_json::json!({
                    "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                    "object": "chat.completion",
                    "created": std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "message": message,
                        "finish_reason": if has_tool_calls { "tool_calls" } else { "stop" }
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }))
            } else {
                let body = resp.text().await.unwrap_or_default();
                if let Some(db) = &state.db {
                    let _ = state
                        .pool_service
                        .mark_unhealthy(db, &credential.uuid, Some(&body));
                }
                Err(format!("Upstream error: {}", body))
            }
        }
        CredentialData::OpenAIKey { api_key, base_url } => {
            let provider = OpenAICustomProvider::with_config(api_key.clone(), base_url.clone());
            let resp = match provider.call_api(request).await {
                Ok(r) => r,
                Err(e) => {
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                    }
                    return Err(e.to_string());
                }
            };
            if resp.status().is_success() {
                // 记录成功
                if let Some(db) = &state.db {
                    let _ =
                        state
                            .pool_service
                            .mark_healthy(db, &credential.uuid, Some(&request.model));
                    let _ = state.pool_service.record_usage(db, &credential.uuid);
                }
                resp.json::<serde_json::Value>()
                    .await
                    .map_err(|e| e.to_string())
            } else {
                let body = resp.text().await.unwrap_or_default();
                if let Some(db) = &state.db {
                    let _ = state
                        .pool_service
                        .mark_unhealthy(db, &credential.uuid, Some(&body));
                }
                Err(format!("Upstream error: {}", body))
            }
        }
        CredentialData::ClaudeKey { api_key, base_url } => {
            // 打印 Claude 代理 URL 用于调试
            let actual_base_url = base_url.as_deref().unwrap_or("https://api.anthropic.com");
            tracing::info!(
                "[CLAUDE] 使用 Claude API 代理: base_url={} credential_uuid={}",
                actual_base_url,
                &credential.uuid[..8]
            );
            let provider = ClaudeCustomProvider::with_config(api_key.clone(), base_url.clone());
            match provider.call_openai_api(request).await {
                Ok(result) => {
                    // 记录成功
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_healthy(
                            db,
                            &credential.uuid,
                            Some(&request.model),
                        );
                        let _ = state.pool_service.record_usage(db, &credential.uuid);
                    }
                    Ok(result)
                }
                Err(e) => {
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                    }
                    Err(e.to_string())
                }
            }
        }
        CredentialData::AntigravityOAuth {
            creds_file_path,
            project_id,
        } => {
            let mut antigravity = AntigravityProvider::new();
            if let Err(e) = antigravity
                .load_credentials_from_path(creds_file_path)
                .await
            {
                if let Some(db) = &state.db {
                    let _ = state.pool_service.mark_unhealthy(
                        db,
                        &credential.uuid,
                        Some(&format!("Failed to load credentials: {}", e)),
                    );
                }
                return Err(e.to_string());
            }

            // 使用新的 validate_token() 方法检查 Token 状态
            let validation_result = antigravity.validate_token();
            tracing::info!("[Antigravity WS] Token 验证结果: {:?}", validation_result);

            // 根据验证结果决定是否刷新
            if validation_result.needs_refresh() {
                tracing::info!("[Antigravity WS] Token 需要刷新，开始刷新...");
                match antigravity.refresh_token_with_retry(3).await {
                    Ok(new_token) => {
                        tracing::info!(
                            "[Antigravity WS] Token 刷新成功，新 token 长度: {}",
                            new_token.len()
                        );
                        // 刷新成功，标记为健康
                        if let Some(db) = &state.db {
                            let _ = state.pool_service.mark_healthy(db, &credential.uuid, None);
                        }
                    }
                    Err(refresh_error) => {
                        tracing::error!("[Antigravity WS] Token 刷新失败: {:?}", refresh_error);
                        // 使用新的 mark_unhealthy_with_details 方法
                        if let Some(db) = &state.db {
                            let _ = state.pool_service.mark_unhealthy_with_details(
                                db,
                                &credential.uuid,
                                &refresh_error,
                            );
                        }
                        return Err(refresh_error.user_message());
                    }
                }
            }

            // 设置项目 ID
            if let Some(pid) = project_id {
                antigravity.project_id = Some(pid.clone());
            }
            let proj_id = antigravity.project_id.clone().unwrap_or_default();

            let antigravity_request = convert_openai_to_antigravity_with_context(request, &proj_id);
            match antigravity
                .call_api("generateContent", &antigravity_request)
                .await
            {
                Ok(resp) => {
                    // 记录成功
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_healthy(
                            db,
                            &credential.uuid,
                            Some(&request.model),
                        );
                        let _ = state.pool_service.record_usage(db, &credential.uuid);
                    }
                    Ok(convert_antigravity_to_openai_response(
                        &resp,
                        &request.model,
                    ))
                }
                Err(e) => {
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                    }
                    Err(e.to_string())
                }
            }
        }
        // GeminiOAuth 和 QwenOAuth 暂不支持 WebSocket，需要使用 HTTP 端点
        _ => Err(
            "This credential type is not yet supported via WebSocket. Please use HTTP endpoints."
                .to_string(),
        ),
    }
}

/// WebSocket 专用的 Anthropic 格式 Provider 调用
pub async fn call_provider_anthropic_for_ws(
    state: &AppState,
    credential: &ProviderCredential,
    request: &AnthropicMessagesRequest,
) -> Result<serde_json::Value, String> {
    use crate::models::provider_pool_model::CredentialData;

    match &credential.credential {
        CredentialData::ClaudeKey { api_key, base_url } => {
            // 打印 Claude 代理 URL 用于调试
            let actual_base_url = base_url.as_deref().unwrap_or("https://api.anthropic.com");
            tracing::info!(
                "[CLAUDE] 使用 Claude API 代理: base_url={} credential_uuid={}",
                actual_base_url,
                &credential.uuid[..8]
            );
            let provider = ClaudeCustomProvider::with_config(api_key.clone(), base_url.clone());
            let resp = match provider.call_api(request).await {
                Ok(r) => r,
                Err(e) => {
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                    }
                    return Err(e.to_string());
                }
            };
            if resp.status().is_success() {
                // 记录成功
                if let Some(db) = &state.db {
                    let _ =
                        state
                            .pool_service
                            .mark_healthy(db, &credential.uuid, Some(&request.model));
                    let _ = state.pool_service.record_usage(db, &credential.uuid);
                }
                resp.json::<serde_json::Value>()
                    .await
                    .map_err(|e| e.to_string())
            } else {
                let body = resp.text().await.unwrap_or_default();
                if let Some(db) = &state.db {
                    let _ = state
                        .pool_service
                        .mark_unhealthy(db, &credential.uuid, Some(&body));
                }
                Err(format!("Upstream error: {}", body))
            }
        }
        _ => {
            // 转换为 OpenAI 格式并调用（健康状态更新在 call_provider_openai_for_ws 中处理）
            let openai_request = convert_anthropic_to_openai(request);
            let result = call_provider_openai_for_ws(state, credential, &openai_request).await?;

            // 转换响应为 Anthropic 格式
            Ok(serde_json::json!({
                "id": format!("msg_{}", uuid::Uuid::new_v4()),
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": result.get("choices")
                        .and_then(|c| c.get(0))
                        .and_then(|c| c.get("message"))
                        .and_then(|m| m.get("content"))
                        .and_then(|c| c.as_str())
                        .unwrap_or("")
                }],
                "model": request.model,
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0
                }
            }))
        }
    }
}
