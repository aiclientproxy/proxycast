//! API 端点处理器
//!
//! 处理 OpenAI 和 Anthropic 格式的 API 请求
//!
//! # 流式传输支持

#![allow(dead_code)]
//!
//! 本模块支持真正的端到端流式传输：
//! - 对于流式请求，使用 StreamManager 处理响应
//! - 集成 Flow Monitor 实时捕获流式内容
//!
//! # 需求覆盖
//!
//! - 需求 5.1: 在收到 chunk 后立即转发给客户端
//! - 需求 5.3: 流中发生错误时发送错误事件并优雅关闭流

use axum::{
    body::Body,
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use chrono::Utc;
use std::collections::HashMap;

use crate::converter::anthropic_to_openai::convert_anthropic_to_openai;
use crate::flow_monitor::{
    ClientInfo, FlowError, FlowErrorType, FlowMetadata, FlowType, InterceptAction, InterceptType,
    LLMFlow, LLMRequest, LLMResponse, Message, MessageContent, MessageRole, RequestParameters,
    RoutingInfo, TokenUsage,
};
use crate::models::anthropic::AnthropicMessagesRequest;
use crate::models::openai::ChatCompletionRequest;
use crate::processor::RequestContext;
use crate::server::client_detector::ClientType;
use crate::server::{record_request_telemetry, record_token_usage, AppState};
use crate::server_utils::{
    build_anthropic_response, build_anthropic_stream_response, message_content_len,
    parse_cw_response, safe_truncate,
};
use crate::streaming::StreamFormat as StreamingFormat;
use crate::ProviderType;

use super::{call_provider_anthropic, call_provider_openai};

// ============================================================================
// Flow 捕获辅助函数
// ============================================================================

/// 从 OpenAI 格式请求构建 LLMRequest
fn build_llm_request_from_openai(
    request: &ChatCompletionRequest,
    path: &str,
    headers: &HeaderMap,
) -> LLMRequest {
    // 转换消息
    let messages: Vec<Message> = request
        .messages
        .iter()
        .map(|m| {
            let role = match m.role.as_str() {
                "system" => MessageRole::System,
                "user" => MessageRole::User,
                "assistant" => MessageRole::Assistant,
                "tool" => MessageRole::Tool,
                "function" => MessageRole::Function,
                _ => MessageRole::User,
            };

            let content = match &m.content {
                Some(c) => match c {
                    crate::models::openai::MessageContent::Text(s) => {
                        MessageContent::Text(s.clone())
                    }
                    crate::models::openai::MessageContent::Parts(parts) => {
                        let flow_parts: Vec<crate::flow_monitor::ContentPart> = parts
                            .iter()
                            .map(|p| match p {
                                crate::models::openai::ContentPart::Text { text } => {
                                    crate::flow_monitor::ContentPart::Text { text: text.clone() }
                                }
                                crate::models::openai::ContentPart::ImageUrl { image_url } => {
                                    crate::flow_monitor::ContentPart::ImageUrl {
                                        image_url: crate::flow_monitor::models::ImageUrl {
                                            url: image_url.url.clone(),
                                            detail: image_url.detail.clone(),
                                        },
                                    }
                                }
                            })
                            .collect();
                        MessageContent::MultiModal(flow_parts)
                    }
                },
                None => MessageContent::Text(String::new()),
            };

            Message {
                role,
                content,
                tool_calls: None,
                tool_result: None,
                name: None,
            }
        })
        .collect();

    // 提取系统提示词
    let system_prompt = messages
        .iter()
        .find(|m| m.role == MessageRole::System)
        .map(|m| m.content.get_all_text());

    // 构建请求参数
    let parameters = RequestParameters {
        temperature: request.temperature,
        top_p: None,
        max_tokens: request.max_tokens,
        stop: None,
        stream: request.stream,
        extra: HashMap::new(),
    };

    // 提取请求头
    let mut header_map = HashMap::new();
    for (name, value) in headers.iter() {
        if let Ok(v) = value.to_str() {
            // 排除敏感头
            let name_lower = name.as_str().to_lowercase();
            if !name_lower.contains("authorization") && !name_lower.contains("api-key") {
                header_map.insert(name.as_str().to_string(), v.to_string());
            }
        }
    }

    LLMRequest {
        method: "POST".to_string(),
        path: path.to_string(),
        headers: header_map,
        body: serde_json::to_value(request).unwrap_or_default(),
        messages,
        system_prompt,
        tools: None, // TODO: 转换工具定义
        model: request.model.clone(),
        original_model: None,
        parameters,
        size_bytes: 0,
        timestamp: Utc::now(),
    }
}

/// 从 Anthropic 格式请求构建 LLMRequest
fn build_llm_request_from_anthropic(
    request: &AnthropicMessagesRequest,
    path: &str,
    headers: &HeaderMap,
) -> LLMRequest {
    // 转换消息
    let messages: Vec<Message> = request
        .messages
        .iter()
        .map(|m| {
            let role = match m.role.as_str() {
                "user" => MessageRole::User,
                "assistant" => MessageRole::Assistant,
                _ => MessageRole::User,
            };

            let content = match &m.content {
                serde_json::Value::String(s) => MessageContent::Text(s.clone()),
                serde_json::Value::Array(arr) => {
                    let flow_parts: Vec<crate::flow_monitor::ContentPart> = arr
                        .iter()
                        .filter_map(|p| {
                            let part_type = p.get("type").and_then(|t| t.as_str()).unwrap_or("");
                            match part_type {
                                "text" => p.get("text").and_then(|t| t.as_str()).map(|text| {
                                    crate::flow_monitor::ContentPart::Text {
                                        text: text.to_string(),
                                    }
                                }),
                                "image" => {
                                    let source = p.get("source")?;
                                    let media_type = source
                                        .get("media_type")
                                        .and_then(|m| m.as_str())
                                        .map(|s| s.to_string());
                                    let data = source
                                        .get("data")
                                        .and_then(|d| d.as_str())
                                        .map(|s| s.to_string());
                                    Some(crate::flow_monitor::ContentPart::Image {
                                        media_type,
                                        data,
                                        url: None,
                                    })
                                }
                                _ => None,
                            }
                        })
                        .collect();
                    MessageContent::MultiModal(flow_parts)
                }
                _ => MessageContent::Text(String::new()),
            };

            Message {
                role,
                content,
                tool_calls: None,
                tool_result: None,
                name: None,
            }
        })
        .collect();

    // 提取系统提示词
    let system_prompt = request.system.as_ref().map(|s| match s {
        serde_json::Value::String(text) => text.clone(),
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    });

    // 构建请求参数
    let parameters = RequestParameters {
        temperature: request.temperature,
        top_p: None,
        max_tokens: request.max_tokens,
        stop: None,
        stream: request.stream,
        extra: HashMap::new(),
    };

    // 提取请求头
    let mut header_map = HashMap::new();
    for (name, value) in headers.iter() {
        if let Ok(v) = value.to_str() {
            let name_lower = name.as_str().to_lowercase();
            if !name_lower.contains("authorization") && !name_lower.contains("api-key") {
                header_map.insert(name.as_str().to_string(), v.to_string());
            }
        }
    }

    LLMRequest {
        method: "POST".to_string(),
        path: path.to_string(),
        headers: header_map,
        body: serde_json::to_value(request).unwrap_or_default(),
        messages,
        system_prompt,
        tools: None, // TODO: 转换工具定义
        model: request.model.clone(),
        original_model: None,
        parameters,
        size_bytes: 0,
        timestamp: Utc::now(),
    }
}

/// 构建 FlowMetadata
fn build_flow_metadata(
    provider: ProviderType,
    credential_id: Option<&str>,
    credential_name: Option<&str>,
    headers: &HeaderMap,
    request_id: &str,
) -> FlowMetadata {
    // 提取客户端信息
    let client_ip = headers
        .get("x-forwarded-for")
        .or_else(|| headers.get("x-real-ip"))
        .and_then(|v| v.to_str().ok())
        .map(|s| s.split(',').next().unwrap_or("").trim().to_string());

    let user_agent = headers
        .get("user-agent")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    FlowMetadata {
        provider,
        credential_id: credential_id.map(|s| s.to_string()),
        credential_name: credential_name.map(|s| s.to_string()),
        retry_count: 0,
        client_info: ClientInfo {
            ip: client_ip,
            user_agent,
            request_id: Some(request_id.to_string()),
        },
        routing_info: RoutingInfo::default(),
        injected_params: None,
        context_usage_percentage: None,
    }
}

/// 从响应构建 LLMResponse
fn build_llm_response(status_code: u16, content: &str, usage: Option<(u32, u32)>) -> LLMResponse {
    let now = Utc::now();
    let (input_tokens, output_tokens) = usage.unwrap_or((0, 0));

    LLMResponse {
        status_code,
        status_text: if status_code == 200 { "OK" } else { "Error" }.to_string(),
        headers: HashMap::new(),
        body: serde_json::Value::Null,
        content: content.to_string(),
        thinking: None,
        tool_calls: Vec::new(),
        usage: TokenUsage {
            input_tokens,
            output_tokens,
            cache_read_tokens: None,
            cache_write_tokens: None,
            thinking_tokens: None,
            total_tokens: input_tokens + output_tokens,
        },
        stop_reason: None,
        size_bytes: content.len(),
        timestamp_start: now,
        timestamp_end: now,
        stream_info: None,
    }
}

// ============================================================================
// Provider 选择辅助函数
// ============================================================================

/// 根据客户端类型和端点配置选择 Provider
///
/// **Validates: Requirements 1.3, 1.4, 3.4**
///
/// 优先级：端点 Provider 配置 > 默认 Provider
///
/// # 参数
/// - `headers`: HTTP 请求头，用于提取 User-Agent
/// - `state`: 应用状态，包含端点配置和默认 Provider
///
/// # 返回
/// 选择的 Provider 名称和检测到的客户端类型
async fn select_provider_for_client(headers: &HeaderMap, state: &AppState) -> (String, ClientType) {
    // 从 User-Agent 检测客户端类型
    let user_agent = headers
        .get("user-agent")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let client_type = ClientType::from_user_agent(user_agent);

    // 获取端点 Provider 配置
    let endpoint_providers = state.endpoint_providers.read().await;
    let endpoint_provider = endpoint_providers.get_provider(client_type.config_key());

    // 获取默认 Provider
    let default_provider = state.default_provider.read().await.clone();

    // 选择 Provider：端点配置优先，否则使用默认
    let selected_provider = match endpoint_provider {
        Some(provider) => provider.clone(),
        None => default_provider,
    };

    (selected_provider, client_type)
}

// ============================================================================
// 拦截检查辅助函数
// ============================================================================

/// 拦截检查结果
pub enum InterceptCheckResult {
    /// 继续处理（可能带有修改后的请求）
    Continue(Option<LLMRequest>),
    /// 请求被取消
    Cancelled,
}

/// 检查是否需要拦截请求
///
/// **Validates: Requirements 2.1, 2.3, 2.5**
///
/// 如果拦截器启用且请求匹配拦截规则，则拦截请求并等待用户操作。
/// 返回 `InterceptCheckResult::Continue` 表示继续处理（可能带有修改后的请求），
/// 返回 `InterceptCheckResult::Cancelled` 表示请求被取消。
async fn check_request_intercept(
    state: &AppState,
    flow_id: &str,
    llm_request: &LLMRequest,
    flow_metadata: &FlowMetadata,
) -> InterceptCheckResult {
    // 创建临时 Flow 用于拦截检查
    let temp_flow = LLMFlow::new(
        flow_id.to_string(),
        FlowType::ChatCompletions,
        llm_request.clone(),
        flow_metadata.clone(),
    );

    // 检查是否需要拦截
    if !state
        .flow_interceptor
        .should_intercept(&temp_flow, &InterceptType::Request)
        .await
    {
        return InterceptCheckResult::Continue(None);
    }

    state.logs.write().await.add(
        "info",
        &format!("[INTERCEPT] 拦截请求: flow_id={}", flow_id),
    );

    // 拦截请求
    let _intercepted = state
        .flow_interceptor
        .intercept_request(flow_id, llm_request.clone())
        .await;

    // 等待用户操作
    let action = state.flow_interceptor.wait_for_action(flow_id).await;

    match action {
        InterceptAction::Continue(modified) => {
            state.logs.write().await.add(
                "info",
                &format!(
                    "[INTERCEPT] 继续处理请求: flow_id={}, modified={}",
                    flow_id,
                    modified.is_some()
                ),
            );
            // 如果有修改，提取修改后的请求
            if let Some(crate::flow_monitor::ModifiedData::Request(req)) = modified {
                InterceptCheckResult::Continue(Some(req))
            } else {
                InterceptCheckResult::Continue(None)
            }
        }
        InterceptAction::Cancel => {
            state.logs.write().await.add(
                "info",
                &format!("[INTERCEPT] 请求被取消: flow_id={}", flow_id),
            );
            InterceptCheckResult::Cancelled
        }
        InterceptAction::Timeout(timeout_action) => {
            state.logs.write().await.add(
                "warn",
                &format!(
                    "[INTERCEPT] 请求超时: flow_id={}, action={:?}",
                    flow_id, timeout_action
                ),
            );
            match timeout_action {
                crate::flow_monitor::TimeoutAction::Continue => {
                    InterceptCheckResult::Continue(None)
                }
                crate::flow_monitor::TimeoutAction::Cancel => InterceptCheckResult::Cancelled,
            }
        }
    }
}

/// 检查是否需要拦截响应
///
/// **Validates: Requirements 2.1, 2.5**
///
/// 如果拦截器启用且响应匹配拦截规则，则拦截响应并等待用户操作。
/// 返回修改后的响应（如果有）或 None。
async fn check_response_intercept(
    state: &AppState,
    flow_id: &str,
    llm_response: &LLMResponse,
    llm_request: &LLMRequest,
    flow_metadata: &FlowMetadata,
) -> Option<LLMResponse> {
    // 创建临时 Flow 用于拦截检查
    let mut temp_flow = LLMFlow::new(
        flow_id.to_string(),
        FlowType::ChatCompletions,
        llm_request.clone(),
        flow_metadata.clone(),
    );
    temp_flow.response = Some(llm_response.clone());

    // 检查是否需要拦截
    if !state
        .flow_interceptor
        .should_intercept(&temp_flow, &InterceptType::Response)
        .await
    {
        return None;
    }

    state.logs.write().await.add(
        "info",
        &format!("[INTERCEPT] 拦截响应: flow_id={}", flow_id),
    );

    // 拦截响应
    let _intercepted = state
        .flow_interceptor
        .intercept_response(flow_id, llm_response.clone())
        .await;

    // 等待用户操作
    let action = state.flow_interceptor.wait_for_action(flow_id).await;

    match action {
        InterceptAction::Continue(modified) => {
            state.logs.write().await.add(
                "info",
                &format!(
                    "[INTERCEPT] 继续处理响应: flow_id={}, modified={}",
                    flow_id,
                    modified.is_some()
                ),
            );
            // 如果有修改，提取修改后的响应
            if let Some(crate::flow_monitor::ModifiedData::Response(resp)) = modified {
                Some(resp)
            } else {
                None
            }
        }
        InterceptAction::Cancel | InterceptAction::Timeout(_) => {
            state.logs.write().await.add(
                "warn",
                &format!("[INTERCEPT] 响应处理被取消或超时: flow_id={}", flow_id),
            );
            None
        }
    }
}

// ============================================================================
// API Key 验证
// ============================================================================

/// OpenAI 格式的 API key 验证
pub async fn verify_api_key(
    headers: &HeaderMap,
    expected_key: &str,
) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    let auth = headers
        .get("authorization")
        .or_else(|| headers.get("x-api-key"))
        .and_then(|v| v.to_str().ok());

    let key = match auth {
        Some(s) if s.starts_with("Bearer ") => &s[7..],
        Some(s) => s,
        None => {
            return Err((
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": {"message": "No API key provided"}})),
            ))
        }
    };

    if key != expected_key {
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error": {"message": "Invalid API key"}})),
        ));
    }

    Ok(())
}

/// Anthropic 格式的 API key 验证
pub async fn verify_api_key_anthropic(
    headers: &HeaderMap,
    expected_key: &str,
) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    let auth = headers
        .get("x-api-key")
        .or_else(|| headers.get("authorization"))
        .and_then(|v| v.to_str().ok());

    let key = match auth {
        Some(s) if s.starts_with("Bearer ") => &s[7..],
        Some(s) => s,
        None => {
            return Err((
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({
                    "type": "error",
                    "error": {
                        "type": "authentication_error",
                        "message": "No API key provided. Please set the x-api-key header."
                    }
                })),
            ))
        }
    };

    if key != expected_key {
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "Invalid API key"
                }
            })),
        ));
    }

    Ok(())
}

pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut request): Json<ChatCompletionRequest>,
) -> Response {
    // ========== 详细日志：请求入口 ==========
    eprintln!("\n========== [CHAT_COMPLETIONS] 收到请求 ==========");
    eprintln!("[CHAT_COMPLETIONS] URL: /v1/chat/completions");
    eprintln!("[CHAT_COMPLETIONS] 模型: {}", request.model);
    eprintln!("[CHAT_COMPLETIONS] 流式: {}", request.stream);
    eprintln!("[CHAT_COMPLETIONS] 消息数量: {}", request.messages.len());

    if let Err(e) = verify_api_key(&headers, &state.api_key).await {
        eprintln!("[CHAT_COMPLETIONS] 认证失败!");
        state
            .logs
            .write()
            .await
            .add("warn", "Unauthorized request to /v1/chat/completions");
        return e.into_response();
    }
    eprintln!("[CHAT_COMPLETIONS] 认证成功");

    // 创建请求上下文
    let mut ctx = RequestContext::new(request.model.clone()).with_stream(request.stream);
    eprintln!("[CHAT_COMPLETIONS] 请求ID: {}", ctx.request_id);

    state.logs.write().await.add(
        "info",
        &format!(
            "POST /v1/chat/completions request_id={} model={} stream={}",
            ctx.request_id, request.model, request.stream
        ),
    );

    // 使用 RequestProcessor 解析模型别名和路由
    eprintln!("[CHAT_COMPLETIONS] 开始路由解析...");
    let provider = state.processor.resolve_and_route(&mut ctx).await;
    eprintln!(
        "[CHAT_COMPLETIONS] 路由结果: provider={:?}, resolved_model={}",
        provider, ctx.resolved_model
    );

    // 更新请求中的模型名为解析后的模型
    if ctx.resolved_model != ctx.original_model {
        request.model = ctx.resolved_model.clone();
        eprintln!(
            "[CHAT_COMPLETIONS] 模型别名解析: {} -> {}",
            ctx.original_model, ctx.resolved_model
        );
        state.logs.write().await.add(
            "info",
            &format!(
                "[MAPPER] request_id={} alias={} -> model={}",
                ctx.request_id, ctx.original_model, ctx.resolved_model
            ),
        );
    }

    // 应用参数注入
    let injection_enabled = *state.injection_enabled.read().await;
    if injection_enabled {
        let injector = state.processor.injector.read().await;
        let mut payload = serde_json::to_value(&request).unwrap_or_default();
        let result = injector.inject(&request.model, &mut payload);
        if result.has_injections() {
            state.logs.write().await.add(
                "info",
                &format!(
                    "[INJECT] request_id={} applied_rules={:?} injected_params={:?}",
                    ctx.request_id, result.applied_rules, result.injected_params
                ),
            );
            // 更新请求
            if let Ok(updated) = serde_json::from_value(payload) {
                request = updated;
            }
        }
    }

    // 根据客户端类型选择 Provider
    // **Validates: Requirements 3.1, 3.3, 3.4**
    let (selected_provider, client_type) = select_provider_for_client(&headers, &state).await;
    eprintln!(
        "[CHAT_COMPLETIONS] 客户端类型: {}, 选择的Provider: {}",
        client_type, selected_provider
    );

    // 记录客户端检测和 Provider 选择结果
    state.logs.write().await.add(
        "info",
        &format!(
            "[CLIENT] request_id={} client_type={} selected_provider={}",
            ctx.request_id, client_type, selected_provider
        ),
    );

    // 记录路由结果
    state.logs.write().await.add(
        "info",
        &format!(
            "[ROUTE] request_id={} model={} provider={}",
            ctx.request_id, ctx.resolved_model, provider
        ),
    );

    // 尝试从凭证池中选择凭证
    // 优先使用路由规则选择的 provider，如果找不到再回退到 selected_provider
    eprintln!("[CHAT_COMPLETIONS] 开始选择凭证...");
    let credential = match &state.db {
        Some(db) => {
            // 首先尝试使用路由规则选择的 provider
            let provider_str = provider.to_string();
            eprintln!(
                "[CHAT_COMPLETIONS] 尝试从凭证池选择: provider={}, model={}",
                provider_str, request.model
            );
            let cred = state
                .pool_service
                .select_credential(db, &provider_str, Some(&request.model))
                .ok()
                .flatten();

            if cred.is_some() {
                eprintln!("[CHAT_COMPLETIONS] 找到凭证: provider={}", provider_str);
            } else {
                eprintln!("[CHAT_COMPLETIONS] 未找到凭证: provider={}", provider_str);
            }

            // 如果路由规则的 provider 没有找到凭证，回退到 selected_provider
            if cred.is_none() && provider_str != selected_provider {
                eprintln!(
                    "[CHAT_COMPLETIONS] 回退到 selected_provider: {}",
                    selected_provider
                );
                state.logs.write().await.add(
                    "debug",
                    &format!(
                        "[ROUTE] No credential found for routed provider '{}', trying selected_provider '{}'",
                        provider_str, selected_provider
                    ),
                );
                let fallback_cred = state
                    .pool_service
                    .select_credential(db, &selected_provider, Some(&request.model))
                    .ok()
                    .flatten();
                if fallback_cred.is_some() {
                    eprintln!(
                        "[CHAT_COMPLETIONS] 回退凭证找到: provider={}",
                        selected_provider
                    );
                } else {
                    eprintln!("[CHAT_COMPLETIONS] 回退凭证也未找到!");
                }
                fallback_cred
            } else {
                cred
            }
        }
        None => {
            eprintln!("[CHAT_COMPLETIONS] 数据库未初始化!");
            None
        }
    };

    // 如果找到凭证池中的凭证，使用它
    if let Some(cred) = credential {
        eprintln!(
            "[CHAT_COMPLETIONS] 使用凭证: type={}, name={:?}, uuid={}",
            cred.provider_type,
            cred.name,
            &cred.uuid[..8.min(cred.uuid.len())]
        );
        state.logs.write().await.add(
            "info",
            &format!(
                "[ROUTE] Using pool credential: type={} name={:?} uuid={}",
                cred.provider_type,
                cred.name,
                &cred.uuid[..8]
            ),
        );

        // 启动 Flow 捕获
        let llm_request = build_llm_request_from_openai(&request, "/v1/chat/completions", &headers);
        let flow_metadata = build_flow_metadata(
            provider,
            Some(&cred.uuid),
            cred.name.as_deref(),
            &headers,
            &ctx.request_id,
        );
        let flow_id = state
            .flow_monitor
            .start_flow(llm_request.clone(), flow_metadata.clone())
            .await;

        // 检查是否需要拦截请求
        // **Validates: Requirements 2.1, 2.3, 2.5**
        if let Some(ref fid) = flow_id {
            match check_request_intercept(&state, fid, &llm_request, &flow_metadata).await {
                InterceptCheckResult::Continue(modified_request) => {
                    // 如果有修改后的请求，更新请求
                    if let Some(modified) = modified_request {
                        // 从修改后的 LLMRequest 更新 ChatCompletionRequest
                        if let Ok(updated) = serde_json::from_value(modified.body.clone()) {
                            request = updated;
                        }
                    }
                }
                InterceptCheckResult::Cancelled => {
                    // 请求被取消，标记 Flow 失败并返回错误
                    let error = FlowError::new(FlowErrorType::Cancelled, "请求被用户取消");
                    state.flow_monitor.fail_flow(fid, error).await;
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(
                            serde_json::json!({"error": {"message": "Request cancelled by user"}}),
                        ),
                    )
                        .into_response();
                }
            }
        }

        eprintln!("[CHAT_COMPLETIONS] 调用 Provider: {}", cred.provider_type);
        let response = call_provider_openai(&state, &cred, &request, flow_id.as_deref()).await;
        eprintln!(
            "[CHAT_COMPLETIONS] Provider 响应状态: {}",
            response.status()
        );

        // 记录请求统计
        let is_success = response.status().is_success();
        let status = if is_success {
            crate::telemetry::RequestStatus::Success
        } else {
            crate::telemetry::RequestStatus::Failed
        };
        record_request_telemetry(&state, &ctx, status, None);

        // 如果成功，记录估算的 Token 使用量
        let estimated_input_tokens = request
            .messages
            .iter()
            .map(|m| {
                let content_len = match &m.content {
                    Some(c) => message_content_len(c),
                    None => 0,
                };
                content_len / 4
            })
            .sum::<usize>() as u32;
        let estimated_output_tokens = if is_success { 100u32 } else { 0u32 };

        if is_success {
            record_token_usage(
                &state,
                &ctx,
                Some(estimated_input_tokens),
                Some(estimated_output_tokens),
            );
        }

        // 完成 Flow 捕获并检查响应拦截
        // **Validates: Requirements 2.1, 2.5**
        if let Some(fid) = flow_id {
            if is_success {
                let llm_response = build_llm_response(
                    200,
                    "", // 内容在 provider_calls 中处理
                    Some((estimated_input_tokens, estimated_output_tokens)),
                );

                // 检查是否需要拦截响应
                if let Some(modified_response) = check_response_intercept(
                    &state,
                    &fid,
                    &llm_response,
                    &llm_request,
                    &flow_metadata,
                )
                .await
                {
                    // 响应被修改，需要重新构建响应
                    state
                        .logs
                        .write()
                        .await
                        .add("info", &format!("[INTERCEPT] 响应被修改: flow_id={}", fid));

                    // 使用修改后的响应完成 Flow
                    state
                        .flow_monitor
                        .complete_flow(&fid, Some(modified_response.clone()))
                        .await;

                    // 构建修改后的 HTTP 响应
                    // 注意：这里简化处理，实际应该根据修改后的内容重新构建完整响应
                    return (
                        StatusCode::OK,
                        Json(serde_json::json!({
                            "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                            "object": "chat.completion",
                            "created": std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs(),
                            "model": request.model,
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": modified_response.content
                                },
                                "finish_reason": "stop"
                            }],
                            "usage": {
                                "prompt_tokens": modified_response.usage.input_tokens,
                                "completion_tokens": modified_response.usage.output_tokens,
                                "total_tokens": modified_response.usage.total_tokens
                            }
                        })),
                    )
                        .into_response();
                }

                state
                    .flow_monitor
                    .complete_flow(&fid, Some(llm_response))
                    .await;
            } else {
                let error = FlowError::new(
                    FlowErrorType::from_status_code(response.status().as_u16()),
                    "Request failed",
                )
                .with_status_code(response.status().as_u16());
                state.flow_monitor.fail_flow(&fid, error).await;
            }
        }

        return response;
    }

    // 回退到旧的单凭证模式（仅当选择的 Provider 是 Kiro 时）
    // 如果选择的 Provider 不是 Kiro，且凭证池中没有找到凭证，返回错误
    // **Validates: Requirements 3.2**
    if selected_provider.to_lowercase() != "kiro" {
        state.logs.write().await.add(
            "error",
            &format!(
                "[ROUTE] No pool credential found for '{}' (client_type={}), and legacy mode only supports Kiro",
                selected_provider, client_type
            ),
        );
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": {
                    "message": format!("没有找到可用的 '{}' 凭证。请在凭证池中添加对应的凭证。", selected_provider),
                    "type": "no_credential_error",
                    "code": "no_credential"
                }
            })),
        )
            .into_response();
    }

    state.logs.write().await.add(
        "debug",
        &format!(
            "[ROUTE] No pool credential found for '{}', using legacy mode",
            selected_provider
        ),
    );

    // 启动 Flow 捕获（legacy mode）
    let llm_request = build_llm_request_from_openai(&request, "/v1/chat/completions", &headers);
    let flow_metadata = build_flow_metadata(provider, None, None, &headers, &ctx.request_id);
    let flow_id = state
        .flow_monitor
        .start_flow(llm_request.clone(), flow_metadata.clone())
        .await;

    // 检查是否需要拦截请求（legacy mode）
    // **Validates: Requirements 2.1, 2.3, 2.5**
    if let Some(ref fid) = flow_id {
        match check_request_intercept(&state, fid, &llm_request, &flow_metadata).await {
            InterceptCheckResult::Continue(modified_request) => {
                // 如果有修改后的请求，更新请求
                if let Some(modified) = modified_request {
                    if let Ok(updated) = serde_json::from_value(modified.body.clone()) {
                        request = updated;
                    }
                }
            }
            InterceptCheckResult::Cancelled => {
                // 请求被取消，标记 Flow 失败并返回错误
                let error = FlowError::new(FlowErrorType::Cancelled, "请求被用户取消");
                state.flow_monitor.fail_flow(fid, error).await;
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({"error": {"message": "Request cancelled by user"}})),
                )
                    .into_response();
            }
        }
    }

    // 检查是否需要刷新 token（无 token 或即将过期）
    {
        let _guard = state.kiro_refresh_lock.lock().await;
        let mut kiro = state.kiro.write().await;
        let needs_refresh =
            kiro.credentials.access_token.is_none() || kiro.is_token_expiring_soon();
        if needs_refresh {
            if let Err(e) = kiro.refresh_token().await {
                state
                    .logs
                    .write()
                    .await
                    .add("error", &format!("Token refresh failed: {e}"));
                // 标记 Flow 失败
                if let Some(fid) = &flow_id {
                    let error = FlowError::new(
                        FlowErrorType::Authentication,
                        &format!("Token refresh failed: {e}"),
                    );
                    state.flow_monitor.fail_flow(fid, error).await;
                }
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {e}")}})),
                ).into_response();
            }
        }
    }

    let kiro = state.kiro.read().await;

    match kiro.call_api(&request).await {
        Ok(resp) => {
            let status = resp.status();
            if status.is_success() {
                match resp.text().await {
                    Ok(body) => {
                        let parsed = parse_cw_response(&body);
                        let has_tool_calls = !parsed.tool_calls.is_empty();

                        state.logs.write().await.add(
                            "info",
                            &format!(
                                "Request completed: content_len={}, tool_calls={}",
                                parsed.content.len(),
                                parsed.tool_calls.len()
                            ),
                        );

                        // 构建消息
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

                        // 估算 Token 数量（基于字符数，约 4 字符 = 1 token）
                        let estimated_output_tokens = (parsed.content.len() / 4) as u32;
                        // 估算输入 Token（基于请求消息）
                        let estimated_input_tokens = request
                            .messages
                            .iter()
                            .map(|m| {
                                let content_len = match &m.content {
                                    Some(c) => message_content_len(c),
                                    None => 0,
                                };
                                content_len / 4
                            })
                            .sum::<usize>()
                            as u32;

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
                                "prompt_tokens": estimated_input_tokens,
                                "completion_tokens": estimated_output_tokens,
                                "total_tokens": estimated_input_tokens + estimated_output_tokens
                            }
                        });
                        // 记录成功请求统计
                        record_request_telemetry(
                            &state,
                            &ctx,
                            crate::telemetry::RequestStatus::Success,
                            None,
                        );
                        // 记录 Token 使用量
                        record_token_usage(
                            &state,
                            &ctx,
                            Some(estimated_input_tokens),
                            Some(estimated_output_tokens),
                        );
                        // 完成 Flow 捕获并检查响应拦截
                        // **Validates: Requirements 2.1, 2.5**
                        if let Some(fid) = &flow_id {
                            let llm_response = build_llm_response(
                                200,
                                &parsed.content,
                                Some((estimated_input_tokens, estimated_output_tokens)),
                            );

                            // 检查是否需要拦截响应
                            if let Some(modified_response) = check_response_intercept(
                                &state,
                                fid,
                                &llm_response,
                                &llm_request,
                                &flow_metadata,
                            )
                            .await
                            {
                                // 响应被修改，需要重新构建响应
                                state.logs.write().await.add(
                                    "info",
                                    &format!("[INTERCEPT] 响应被修改: flow_id={}", fid),
                                );

                                // 使用修改后的响应完成 Flow
                                state
                                    .flow_monitor
                                    .complete_flow(fid, Some(modified_response.clone()))
                                    .await;

                                // 构建修改后的响应
                                let modified_message = serde_json::json!({
                                    "role": "assistant",
                                    "content": modified_response.content
                                });

                                let modified_json_response = serde_json::json!({
                                    "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                                    "object": "chat.completion",
                                    "created": std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .unwrap_or_default()
                                        .as_secs(),
                                    "model": request.model,
                                    "choices": [{
                                        "index": 0,
                                        "message": modified_message,
                                        "finish_reason": "stop"
                                    }],
                                    "usage": {
                                        "prompt_tokens": modified_response.usage.input_tokens,
                                        "completion_tokens": modified_response.usage.output_tokens,
                                        "total_tokens": modified_response.usage.total_tokens
                                    }
                                });

                                return Json(modified_json_response).into_response();
                            }

                            state
                                .flow_monitor
                                .complete_flow(fid, Some(llm_response))
                                .await;
                        }
                        Json(response).into_response()
                    }
                    Err(e) => {
                        // 记录失败请求统计
                        record_request_telemetry(
                            &state,
                            &ctx,
                            crate::telemetry::RequestStatus::Failed,
                            Some(e.to_string()),
                        );
                        // 标记 Flow 失败
                        if let Some(fid) = &flow_id {
                            let error = FlowError::new(FlowErrorType::Network, &e.to_string());
                            state.flow_monitor.fail_flow(fid, error).await;
                        }
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": e.to_string()}})),
                        )
                            .into_response()
                    }
                }
            } else if status.as_u16() == 403 || status.as_u16() == 402 {
                // Token 过期或账户问题，尝试重新加载凭证并刷新
                drop(kiro);
                let _guard = state.kiro_refresh_lock.lock().await;
                let mut kiro = state.kiro.write().await;
                state.logs.write().await.add(
                    "warn",
                    &format!(
                        "[AUTH] Got {}, reloading credentials and attempting token refresh...",
                        status.as_u16()
                    ),
                );

                // 先重新加载凭证文件（可能用户换了账户）
                if let Err(e) = kiro.load_credentials().await {
                    state.logs.write().await.add(
                        "error",
                        &format!("[AUTH] Failed to reload credentials: {e}"),
                    );
                }

                match kiro.refresh_token().await {
                    Ok(_) => {
                        state
                            .logs
                            .write()
                            .await
                            .add("info", "[AUTH] Token refreshed successfully after reload");
                        // 重试请求
                        drop(kiro);
                        let kiro = state.kiro.read().await;
                        match kiro.call_api(&request).await {
                            Ok(retry_resp) => {
                                if retry_resp.status().is_success() {
                                    match retry_resp.text().await {
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
                                            // 完成 Flow 捕获并检查响应拦截（重试成功）
                                            // **Validates: Requirements 2.1, 2.5**
                                            if let Some(fid) = &flow_id {
                                                let (est_input, est_output) =
                                                    parsed.estimate_tokens();
                                                let llm_response = build_llm_response(
                                                    200,
                                                    &parsed.content,
                                                    Some((est_input, est_output)),
                                                );

                                                // 检查是否需要拦截响应
                                                if let Some(modified_response) =
                                                    check_response_intercept(
                                                        &state,
                                                        fid,
                                                        &llm_response,
                                                        &llm_request,
                                                        &flow_metadata,
                                                    )
                                                    .await
                                                {
                                                    // 响应被修改，需要重新构建响应
                                                    state.logs.write().await.add(
                                                        "info",
                                                        &format!(
                                                            "[INTERCEPT] 响应被修改: flow_id={}",
                                                            fid
                                                        ),
                                                    );

                                                    // 使用修改后的响应完成 Flow
                                                    state
                                                        .flow_monitor
                                                        .complete_flow(
                                                            fid,
                                                            Some(modified_response.clone()),
                                                        )
                                                        .await;

                                                    // 构建修改后的响应
                                                    let modified_message = serde_json::json!({
                                                        "role": "assistant",
                                                        "content": modified_response.content
                                                    });

                                                    let modified_json_response = serde_json::json!({
                                                        "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                                                        "object": "chat.completion",
                                                        "created": std::time::SystemTime::now()
                                                            .duration_since(std::time::UNIX_EPOCH)
                                                            .unwrap_or_default()
                                                            .as_secs(),
                                                        "model": request.model,
                                                        "choices": [{
                                                            "index": 0,
                                                            "message": modified_message,
                                                            "finish_reason": "stop"
                                                        }],
                                                        "usage": {
                                                            "prompt_tokens": modified_response.usage.input_tokens,
                                                            "completion_tokens": modified_response.usage.output_tokens,
                                                            "total_tokens": modified_response.usage.total_tokens
                                                        }
                                                    });

                                                    return Json(modified_json_response)
                                                        .into_response();
                                                }

                                                state
                                                    .flow_monitor
                                                    .complete_flow(fid, Some(llm_response))
                                                    .await;
                                            }
                                            return Json(response).into_response();
                                        }
                                        Err(e) => {
                                            // 标记 Flow 失败
                                            if let Some(fid) = &flow_id {
                                                let error = FlowError::new(
                                                    FlowErrorType::Network,
                                                    &e.to_string(),
                                                );
                                                state.flow_monitor.fail_flow(fid, error).await;
                                            }
                                            return (
                                            StatusCode::INTERNAL_SERVER_ERROR,
                                            Json(serde_json::json!({"error": {"message": e.to_string()}})),
                                        ).into_response();
                                        }
                                    }
                                }
                                let body = retry_resp.text().await.unwrap_or_default();
                                // 标记 Flow 失败（重试失败）
                                if let Some(fid) = &flow_id {
                                    let error = FlowError::new(
                                        FlowErrorType::ServerError,
                                        &format!("Retry failed: {}", body),
                                    );
                                    state.flow_monitor.fail_flow(fid, error).await;
                                }
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    Json(serde_json::json!({"error": {"message": format!("Retry failed: {}", body)}})),
                                ).into_response()
                            }
                            Err(e) => {
                                // 标记 Flow 失败
                                if let Some(fid) = &flow_id {
                                    let error =
                                        FlowError::new(FlowErrorType::Network, &e.to_string());
                                    state.flow_monitor.fail_flow(fid, error).await;
                                }
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    Json(serde_json::json!({"error": {"message": e.to_string()}})),
                                )
                                    .into_response()
                            }
                        }
                    }
                    Err(e) => {
                        state
                            .logs
                            .write()
                            .await
                            .add("error", &format!("[AUTH] Token refresh failed: {e}"));
                        // 标记 Flow 失败
                        if let Some(fid) = &flow_id {
                            let error = FlowError::new(
                                FlowErrorType::Authentication,
                                &format!("Token refresh failed: {e}"),
                            );
                            state.flow_monitor.fail_flow(fid, error).await;
                        }
                        (
                            StatusCode::UNAUTHORIZED,
                            Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {e}")}})),
                        )
                            .into_response()
                    }
                }
            } else {
                let body = resp.text().await.unwrap_or_default();
                state.logs.write().await.add(
                    "error",
                    &format!("Upstream error {}: {}", status, safe_truncate(&body, 200)),
                );
                // 标记 Flow 失败
                if let Some(fid) = &flow_id {
                    let error =
                        FlowError::new(FlowErrorType::from_status_code(status.as_u16()), &body)
                            .with_status_code(status.as_u16());
                    state.flow_monitor.fail_flow(fid, error).await;
                }
                (
                    StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                    Json(serde_json::json!({"error": {"message": format!("Upstream error: {}", body)}}))
                ).into_response()
            }
        }
        Err(e) => {
            state
                .logs
                .write()
                .await
                .add("error", &format!("API call failed: {e}"));
            // 标记 Flow 失败
            if let Some(fid) = &flow_id {
                let error = FlowError::new(FlowErrorType::Network, &e.to_string());
                state.flow_monitor.fail_flow(fid, error).await;
            }
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": {"message": e.to_string()}})),
            )
                .into_response()
        }
    }
}

pub async fn anthropic_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut request): Json<AnthropicMessagesRequest>,
) -> Response {
    // 使用 Anthropic 格式的认证验证（优先检查 x-api-key）
    if let Err(e) = verify_api_key_anthropic(&headers, &state.api_key).await {
        state
            .logs
            .write()
            .await
            .add("warn", "Unauthorized request to /v1/messages");
        return e.into_response();
    }

    // 创建请求上下文
    let mut ctx = RequestContext::new(request.model.clone()).with_stream(request.stream);

    // 详细记录请求信息
    let msg_count = request.messages.len();
    let has_tools = request.tools.as_ref().map(|t| t.len()).unwrap_or(0);
    let has_system = request.system.is_some();
    state.logs.write().await.add(
        "info",
        &format!(
            "[REQ] POST /v1/messages request_id={} model={} stream={} messages={} tools={} has_system={}",
            ctx.request_id, request.model, request.stream, msg_count, has_tools, has_system
        ),
    );

    // 使用 RequestProcessor 解析模型别名和路由
    let provider = state.processor.resolve_and_route(&mut ctx).await;

    // 更新请求中的模型名为解析后的模型
    if ctx.resolved_model != ctx.original_model {
        request.model = ctx.resolved_model.clone();
        state.logs.write().await.add(
            "info",
            &format!(
                "[MAPPER] request_id={} alias={} -> model={}",
                ctx.request_id, ctx.original_model, ctx.resolved_model
            ),
        );
    }

    // 记录最后一条消息的角色和内容预览
    if let Some(last_msg) = request.messages.last() {
        let content_preview = match &last_msg.content {
            serde_json::Value::String(s) => s.chars().take(100).collect::<String>(),
            serde_json::Value::Array(arr) => {
                if let Some(first) = arr.first() {
                    if let Some(text) = first.get("text").and_then(|t| t.as_str()) {
                        text.chars().take(100).collect::<String>()
                    } else {
                        format!("[{} blocks]", arr.len())
                    }
                } else {
                    "[empty]".to_string()
                }
            }
            _ => "[unknown]".to_string(),
        };
        state.logs.write().await.add(
            "debug",
            &format!(
                "[REQ] request_id={} last_message: role={} content={}",
                ctx.request_id, last_msg.role, content_preview
            ),
        );
    }

    // 应用参数注入
    let injection_enabled = *state.injection_enabled.read().await;
    if injection_enabled {
        let injector = state.processor.injector.read().await;
        let mut payload = serde_json::to_value(&request).unwrap_or_default();
        let result = injector.inject(&request.model, &mut payload);
        if result.has_injections() {
            state.logs.write().await.add(
                "info",
                &format!(
                    "[INJECT] request_id={} applied_rules={:?} injected_params={:?}",
                    ctx.request_id, result.applied_rules, result.injected_params
                ),
            );
            // 更新请求
            if let Ok(updated) = serde_json::from_value(payload) {
                request = updated;
            }
        }
    }

    // 根据客户端类型选择 Provider
    // **Validates: Requirements 3.1, 3.3, 3.4**
    let (selected_provider, client_type) = select_provider_for_client(&headers, &state).await;

    // 记录客户端检测和 Provider 选择结果
    state.logs.write().await.add(
        "info",
        &format!(
            "[CLIENT] request_id={} client_type={} selected_provider={}",
            ctx.request_id, client_type, selected_provider
        ),
    );

    // 记录路由结果
    state.logs.write().await.add(
        "info",
        &format!(
            "[ROUTE] request_id={} model={} provider={}",
            ctx.request_id, ctx.resolved_model, provider
        ),
    );

    // 尝试从凭证池中选择凭证
    let credential = match &state.db {
        Some(db) => {
            // 根据选择的 Provider 配置选择凭证
            state
                .pool_service
                .select_credential(db, &selected_provider, Some(&request.model))
                .ok()
                .flatten()
        }
        None => None,
    };

    // 如果找到凭证池中的凭证，使用它
    if let Some(cred) = credential {
        state.logs.write().await.add(
            "info",
            &format!(
                "[ROUTE] Using pool credential: type={} name={:?} uuid={}",
                cred.provider_type,
                cred.name,
                &cred.uuid[..8]
            ),
        );

        // 启动 Flow 捕获
        let llm_request = build_llm_request_from_anthropic(&request, "/v1/messages", &headers);
        let flow_metadata = build_flow_metadata(
            provider,
            Some(&cred.uuid),
            cred.name.as_deref(),
            &headers,
            &ctx.request_id,
        );
        let flow_id = state
            .flow_monitor
            .start_flow(llm_request.clone(), flow_metadata.clone())
            .await;

        // 检查是否需要拦截请求
        // **Validates: Requirements 2.1, 2.3, 2.5**
        if let Some(ref fid) = flow_id {
            match check_request_intercept(&state, fid, &llm_request, &flow_metadata).await {
                InterceptCheckResult::Continue(modified_request) => {
                    // 如果有修改后的请求，更新请求
                    if let Some(modified) = modified_request {
                        // 从修改后的 LLMRequest 更新 AnthropicMessagesRequest
                        if let Ok(updated) = serde_json::from_value(modified.body.clone()) {
                            request = updated;
                        }
                    }
                }
                InterceptCheckResult::Cancelled => {
                    // 请求被取消，标记 Flow 失败并返回错误
                    let error = FlowError::new(FlowErrorType::Cancelled, "请求被用户取消");
                    state.flow_monitor.fail_flow(fid, error).await;
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({
                            "type": "error",
                            "error": {
                                "type": "request_cancelled",
                                "message": "Request cancelled by user"
                            }
                        })),
                    )
                        .into_response();
                }
            }
        }

        let response = call_provider_anthropic(&state, &cred, &request, flow_id.as_deref()).await;

        // 记录请求统计
        let is_success = response.status().is_success();
        let status = if is_success {
            crate::telemetry::RequestStatus::Success
        } else {
            crate::telemetry::RequestStatus::Failed
        };
        record_request_telemetry(&state, &ctx, status, None);

        // 估算 Token 使用量
        let estimated_input_tokens = request
            .messages
            .iter()
            .map(|m| {
                let content_len = match &m.content {
                    serde_json::Value::String(s) => s.len(),
                    serde_json::Value::Array(arr) => arr
                        .iter()
                        .filter_map(|v| v.get("text").and_then(|t| t.as_str()))
                        .map(|s| s.len())
                        .sum(),
                    _ => 0,
                };
                content_len / 4
            })
            .sum::<usize>() as u32;
        let estimated_output_tokens = if is_success { 100u32 } else { 0u32 };

        if is_success {
            record_token_usage(
                &state,
                &ctx,
                Some(estimated_input_tokens),
                Some(estimated_output_tokens),
            );
        }

        // 完成 Flow 捕获并检查响应拦截
        // **Validates: Requirements 2.1, 2.5**
        if let Some(fid) = flow_id {
            if is_success {
                let llm_response = build_llm_response(
                    200,
                    "",
                    Some((estimated_input_tokens, estimated_output_tokens)),
                );

                // 检查是否需要拦截响应
                if let Some(modified_response) = check_response_intercept(
                    &state,
                    &fid,
                    &llm_response,
                    &llm_request,
                    &flow_metadata,
                )
                .await
                {
                    // 响应被修改，需要重新构建响应
                    state
                        .logs
                        .write()
                        .await
                        .add("info", &format!("[INTERCEPT] 响应被修改: flow_id={}", fid));

                    // 使用修改后的响应完成 Flow
                    state
                        .flow_monitor
                        .complete_flow(&fid, Some(modified_response.clone()))
                        .await;

                    // 构建修改后的 Anthropic 格式响应
                    return (
                        StatusCode::OK,
                        Json(serde_json::json!({
                            "id": format!("msg_{}", uuid::Uuid::new_v4()),
                            "type": "message",
                            "role": "assistant",
                            "content": [{
                                "type": "text",
                                "text": modified_response.content
                            }],
                            "model": request.model,
                            "stop_reason": "end_turn",
                            "stop_sequence": null,
                            "usage": {
                                "input_tokens": modified_response.usage.input_tokens,
                                "output_tokens": modified_response.usage.output_tokens
                            }
                        })),
                    )
                        .into_response();
                }

                state
                    .flow_monitor
                    .complete_flow(&fid, Some(llm_response))
                    .await;
            } else {
                let error = FlowError::new(
                    FlowErrorType::from_status_code(response.status().as_u16()),
                    "Request failed",
                )
                .with_status_code(response.status().as_u16());
                state.flow_monitor.fail_flow(&fid, error).await;
            }
        }

        return response;
    }

    // 回退到旧的单凭证模式（仅当选择的 Provider 是 Kiro 时）
    // 如果选择的 Provider 不是 Kiro，且凭证池中没有找到凭证，返回错误
    // **Validates: Requirements 3.2**
    if selected_provider.to_lowercase() != "kiro" {
        state.logs.write().await.add(
            "error",
            &format!(
                "[ROUTE] No pool credential found for '{}' (client_type={}), and legacy mode only supports Kiro",
                selected_provider, client_type
            ),
        );
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "type": "error",
                "error": {
                    "type": "no_credential_error",
                    "message": format!("没有找到可用的 '{}' 凭证。请在凭证池中添加对应的凭证。", selected_provider)
                }
            })),
        )
            .into_response();
    }

    state.logs.write().await.add(
        "debug",
        &format!(
            "[ROUTE] No pool credential found for '{}', using legacy mode",
            selected_provider
        ),
    );

    // 启动 Flow 捕获（legacy mode）
    let llm_request = build_llm_request_from_anthropic(&request, "/v1/messages", &headers);
    let flow_metadata = build_flow_metadata(provider, None, None, &headers, &ctx.request_id);
    let flow_id = state
        .flow_monitor
        .start_flow(llm_request.clone(), flow_metadata.clone())
        .await;

    // 检查是否需要拦截请求（legacy mode）
    // **Validates: Requirements 2.1, 2.3, 2.5**
    if let Some(ref fid) = flow_id {
        match check_request_intercept(&state, fid, &llm_request, &flow_metadata).await {
            InterceptCheckResult::Continue(modified_request) => {
                // 如果有修改后的请求，更新请求
                if let Some(modified) = modified_request {
                    if let Ok(updated) = serde_json::from_value(modified.body.clone()) {
                        request = updated;
                    }
                }
            }
            InterceptCheckResult::Cancelled => {
                // 请求被取消，标记 Flow 失败并返回错误
                let error = FlowError::new(FlowErrorType::Cancelled, "请求被用户取消");
                state.flow_monitor.fail_flow(fid, error).await;
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "type": "error",
                        "error": {
                            "type": "request_cancelled",
                            "message": "Request cancelled by user"
                        }
                    })),
                )
                    .into_response();
            }
        }
    }

    // 检查是否需要刷新 token（无 token 或即将过期）
    {
        let _guard = state.kiro_refresh_lock.lock().await;
        let mut kiro = state.kiro.write().await;
        let needs_refresh =
            kiro.credentials.access_token.is_none() || kiro.is_token_expiring_soon();
        if needs_refresh {
            state.logs.write().await.add(
                "info",
                "[AUTH] No access token or token expiring soon, attempting refresh...",
            );
            if let Err(e) = kiro.refresh_token().await {
                state
                    .logs
                    .write()
                    .await
                    .add("error", &format!("[AUTH] Token refresh failed: {e}"));
                // 标记 Flow 失败
                if let Some(fid) = &flow_id {
                    let error = FlowError::new(
                        FlowErrorType::Authentication,
                        &format!("Token refresh failed: {e}"),
                    );
                    state.flow_monitor.fail_flow(fid, error).await;
                }
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {e}")}})),
                )
                    .into_response();
            }
            state
                .logs
                .write()
                .await
                .add("info", "[AUTH] Token refreshed successfully");
        }
    }

    // 转换为 OpenAI 格式
    let openai_request = convert_anthropic_to_openai(&request);

    // 记录转换后的请求信息
    state.logs.write().await.add(
        "debug",
        &format!(
            "[CONVERT] OpenAI format: messages={} tools={} stream={}",
            openai_request.messages.len(),
            openai_request.tools.as_ref().map(|t| t.len()).unwrap_or(0),
            openai_request.stream
        ),
    );

    let kiro = state.kiro.read().await;

    match kiro.call_api(&openai_request).await {
        Ok(resp) => {
            let status = resp.status();
            state
                .logs
                .write()
                .await
                .add("info", &format!("[RESP] Upstream status: {status}"));

            if status.is_success() {
                match resp.bytes().await {
                    Ok(bytes) => {
                        // 使用 lossy 转换，避免无效 UTF-8 导致崩溃
                        let body = String::from_utf8_lossy(&bytes).to_string();

                        // 记录原始响应长度
                        state.logs.write().await.add(
                            "debug",
                            &format!("[RESP] Raw body length: {} bytes", bytes.len()),
                        );

                        // 保存原始响应到文件用于调试
                        let request_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
                        state.logs.read().await.log_raw_response(&request_id, &body);
                        state.logs.write().await.add(
                            "debug",
                            &format!("[RESP] Raw response saved to raw_response_{request_id}.txt"),
                        );

                        // 记录响应的前200字符用于调试（减少日志量）
                        let preview: String =
                            body.chars().filter(|c| !c.is_control()).take(200).collect();
                        state
                            .logs
                            .write()
                            .await
                            .add("debug", &format!("[RESP] Body preview: {preview}"));

                        let parsed = parse_cw_response(&body);

                        // 详细记录解析结果
                        state.logs.write().await.add(
                            "info",
                            &format!(
                                "[RESP] Parsed: content_len={}, tool_calls={}, content_preview={}",
                                parsed.content.len(),
                                parsed.tool_calls.len(),
                                parsed.content.chars().take(100).collect::<String>()
                            ),
                        );

                        // 记录 tool calls 详情
                        for (i, tc) in parsed.tool_calls.iter().enumerate() {
                            state.logs.write().await.add(
                                "debug",
                                &format!(
                                    "[RESP] Tool call {}: name={} id={}",
                                    i, tc.function.name, tc.id
                                ),
                            );
                        }

                        // 如果请求流式响应，返回 SSE 格式
                        if request.stream {
                            // 完成 Flow 捕获并检查响应拦截（流式）
                            // **Validates: Requirements 2.1, 2.5**
                            if let Some(fid) = &flow_id {
                                let (est_input, est_output) = parsed.estimate_tokens();
                                let llm_response = build_llm_response(
                                    200,
                                    &parsed.content,
                                    Some((est_input, est_output)),
                                );

                                // 检查是否需要拦截响应
                                if let Some(modified_response) = check_response_intercept(
                                    &state,
                                    fid,
                                    &llm_response,
                                    &llm_request,
                                    &flow_metadata,
                                )
                                .await
                                {
                                    // 响应被修改，需要重新构建响应
                                    state.logs.write().await.add(
                                        "info",
                                        &format!("[INTERCEPT] 流式响应被修改: flow_id={}", fid),
                                    );

                                    // 使用修改后的响应完成 Flow
                                    state
                                        .flow_monitor
                                        .complete_flow(fid, Some(modified_response.clone()))
                                        .await;

                                    // 构建修改后的流式响应
                                    // 注意：这里简化处理，实际应该构建完整的流式响应
                                    return (
                                        StatusCode::OK,
                                        Json(serde_json::json!({
                                            "id": format!("msg_{}", uuid::Uuid::new_v4()),
                                            "type": "message",
                                            "role": "assistant",
                                            "content": [{
                                                "type": "text",
                                                "text": modified_response.content
                                            }],
                                            "model": request.model,
                                            "stop_reason": "end_turn",
                                            "stop_sequence": null,
                                            "usage": {
                                                "input_tokens": modified_response.usage.input_tokens,
                                                "output_tokens": modified_response.usage.output_tokens
                                            }
                                        })),
                                    )
                                        .into_response();
                                }

                                state
                                    .flow_monitor
                                    .complete_flow(fid, Some(llm_response))
                                    .await;
                            }
                            return build_anthropic_stream_response(&request.model, &parsed);
                        }

                        // 完成 Flow 捕获并检查响应拦截（非流式）
                        // **Validates: Requirements 2.1, 2.5**
                        if let Some(fid) = &flow_id {
                            let (est_input, est_output) = parsed.estimate_tokens();
                            let llm_response = build_llm_response(
                                200,
                                &parsed.content,
                                Some((est_input, est_output)),
                            );

                            // 检查是否需要拦截响应
                            if let Some(modified_response) = check_response_intercept(
                                &state,
                                fid,
                                &llm_response,
                                &llm_request,
                                &flow_metadata,
                            )
                            .await
                            {
                                // 响应被修改，需要重新构建响应
                                state.logs.write().await.add(
                                    "info",
                                    &format!("[INTERCEPT] 响应被修改: flow_id={}", fid),
                                );

                                // 使用修改后的响应完成 Flow
                                state
                                    .flow_monitor
                                    .complete_flow(fid, Some(modified_response.clone()))
                                    .await;

                                // 构建修改后的 Anthropic 格式响应
                                return (
                                    StatusCode::OK,
                                    Json(serde_json::json!({
                                        "id": format!("msg_{}", uuid::Uuid::new_v4()),
                                        "type": "message",
                                        "role": "assistant",
                                        "content": [{
                                            "type": "text",
                                            "text": modified_response.content
                                        }],
                                        "model": request.model,
                                        "stop_reason": "end_turn",
                                        "stop_sequence": null,
                                        "usage": {
                                            "input_tokens": modified_response.usage.input_tokens,
                                            "output_tokens": modified_response.usage.output_tokens
                                        }
                                    })),
                                )
                                    .into_response();
                            }

                            state
                                .flow_monitor
                                .complete_flow(fid, Some(llm_response))
                                .await;
                        }

                        // 非流式响应
                        build_anthropic_response(&request.model, &parsed)
                    }
                    Err(e) => {
                        state
                            .logs
                            .write()
                            .await
                            .add("error", &format!("[ERROR] Response body read failed: {e}"));
                        // 标记 Flow 失败
                        if let Some(fid) = &flow_id {
                            let error = FlowError::new(FlowErrorType::Network, &e.to_string());
                            state.flow_monitor.fail_flow(fid, error).await;
                        }
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": e.to_string()}})),
                        )
                            .into_response()
                    }
                }
            } else if status.as_u16() == 403 || status.as_u16() == 402 {
                // Token 过期或账户问题，尝试重新加载凭证并刷新
                drop(kiro);
                let _guard = state.kiro_refresh_lock.lock().await;
                let mut kiro = state.kiro.write().await;
                state.logs.write().await.add(
                    "warn",
                    &format!(
                        "[AUTH] Got {}, reloading credentials and attempting token refresh...",
                        status.as_u16()
                    ),
                );

                // 先重新加载凭证文件（可能用户换了账户）
                if let Err(e) = kiro.load_credentials().await {
                    state.logs.write().await.add(
                        "error",
                        &format!("[AUTH] Failed to reload credentials: {e}"),
                    );
                }

                match kiro.refresh_token().await {
                    Ok(_) => {
                        state.logs.write().await.add(
                            "info",
                            "[AUTH] Token refreshed successfully, retrying request...",
                        );
                        drop(kiro);
                        let kiro = state.kiro.read().await;
                        match kiro.call_api(&openai_request).await {
                            Ok(retry_resp) => {
                                let retry_status = retry_resp.status();
                                state.logs.write().await.add(
                                    "info",
                                    &format!("[RETRY] Response status: {retry_status}"),
                                );
                                if retry_resp.status().is_success() {
                                    match retry_resp.bytes().await {
                                        Ok(bytes) => {
                                            let body = String::from_utf8_lossy(&bytes).to_string();
                                            let parsed = parse_cw_response(&body);
                                            state.logs.write().await.add(
                                                "info",
                                                &format!(
                                                "[RETRY] Success: content_len={}, tool_calls={}",
                                                parsed.content.len(), parsed.tool_calls.len()
                                            ),
                                            );
                                            // 完成 Flow 捕获并检查响应拦截（重试成功）
                                            // **Validates: Requirements 2.1, 2.5**
                                            if let Some(fid) = &flow_id {
                                                let (est_input, est_output) =
                                                    parsed.estimate_tokens();
                                                let llm_response = build_llm_response(
                                                    200,
                                                    &parsed.content,
                                                    Some((est_input, est_output)),
                                                );

                                                // 检查是否需要拦截响应
                                                if let Some(modified_response) =
                                                    check_response_intercept(
                                                        &state,
                                                        fid,
                                                        &llm_response,
                                                        &llm_request,
                                                        &flow_metadata,
                                                    )
                                                    .await
                                                {
                                                    // 响应被修改，需要重新构建响应
                                                    state.logs.write().await.add(
                                                        "info",
                                                        &format!("[INTERCEPT] 重试响应被修改: flow_id={}", fid),
                                                    );

                                                    // 使用修改后的响应完成 Flow
                                                    state
                                                        .flow_monitor
                                                        .complete_flow(
                                                            fid,
                                                            Some(modified_response.clone()),
                                                        )
                                                        .await;

                                                    // 构建修改后的响应
                                                    if request.stream {
                                                        return (
                                                            StatusCode::OK,
                                                            Json(serde_json::json!({
                                                                "id": format!("msg_{}", uuid::Uuid::new_v4()),
                                                                "type": "message",
                                                                "role": "assistant",
                                                                "content": [{
                                                                    "type": "text",
                                                                    "text": modified_response.content
                                                                }],
                                                                "model": request.model,
                                                                "stop_reason": "end_turn",
                                                                "stop_sequence": null,
                                                                "usage": {
                                                                    "input_tokens": modified_response.usage.input_tokens,
                                                                    "output_tokens": modified_response.usage.output_tokens
                                                                }
                                                            })),
                                                        )
                                                            .into_response();
                                                    } else {
                                                        return (
                                                            StatusCode::OK,
                                                            Json(serde_json::json!({
                                                                "id": format!("msg_{}", uuid::Uuid::new_v4()),
                                                                "type": "message",
                                                                "role": "assistant",
                                                                "content": [{
                                                                    "type": "text",
                                                                    "text": modified_response.content
                                                                }],
                                                                "model": request.model,
                                                                "stop_reason": "end_turn",
                                                                "stop_sequence": null,
                                                                "usage": {
                                                                    "input_tokens": modified_response.usage.input_tokens,
                                                                    "output_tokens": modified_response.usage.output_tokens
                                                                }
                                                            })),
                                                        )
                                                            .into_response();
                                                    }
                                                }

                                                state
                                                    .flow_monitor
                                                    .complete_flow(fid, Some(llm_response))
                                                    .await;
                                            }
                                            if request.stream {
                                                return build_anthropic_stream_response(
                                                    &request.model,
                                                    &parsed,
                                                );
                                            }
                                            return build_anthropic_response(
                                                &request.model,
                                                &parsed,
                                            );
                                        }
                                        Err(e) => {
                                            state.logs.write().await.add(
                                                "error",
                                                &format!("[RETRY] Body read failed: {e}"),
                                            );
                                            // 标记 Flow 失败
                                            if let Some(fid) = &flow_id {
                                                let error = FlowError::new(
                                                    FlowErrorType::Network,
                                                    &e.to_string(),
                                                );
                                                state.flow_monitor.fail_flow(fid, error).await;
                                            }
                                            return (
                                                StatusCode::INTERNAL_SERVER_ERROR,
                                                Json(serde_json::json!({"error": {"message": e.to_string()}})),
                                            )
                                                .into_response();
                                        }
                                    }
                                }
                                let body = retry_resp
                                    .bytes()
                                    .await
                                    .map(|b| String::from_utf8_lossy(&b).to_string())
                                    .unwrap_or_default();
                                state.logs.write().await.add(
                                    "error",
                                    &format!(
                                        "[RETRY] Failed with status {retry_status}: {}",
                                        safe_truncate(&body, 500)
                                    ),
                                );
                                // 标记 Flow 失败（重试失败）
                                if let Some(fid) = &flow_id {
                                    let error = FlowError::new(
                                        FlowErrorType::ServerError,
                                        &format!("Retry failed: {}", body),
                                    );
                                    state.flow_monitor.fail_flow(fid, error).await;
                                }
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    Json(serde_json::json!({"error": {"message": format!("Retry failed: {}", body)}})),
                                )
                                    .into_response()
                            }
                            Err(e) => {
                                state
                                    .logs
                                    .write()
                                    .await
                                    .add("error", &format!("[RETRY] Request failed: {e}"));
                                // 标记 Flow 失败
                                if let Some(fid) = &flow_id {
                                    let error =
                                        FlowError::new(FlowErrorType::Network, &e.to_string());
                                    state.flow_monitor.fail_flow(fid, error).await;
                                }
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    Json(serde_json::json!({"error": {"message": e.to_string()}})),
                                )
                                    .into_response()
                            }
                        }
                    }
                    Err(e) => {
                        state
                            .logs
                            .write()
                            .await
                            .add("error", &format!("[AUTH] Token refresh failed: {e}"));
                        // 标记 Flow 失败
                        if let Some(fid) = &flow_id {
                            let error = FlowError::new(
                                FlowErrorType::Authentication,
                                &format!("Token refresh failed: {e}"),
                            );
                            state.flow_monitor.fail_flow(fid, error).await;
                        }
                        (
                            StatusCode::UNAUTHORIZED,
                            Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {e}")}})),
                        )
                            .into_response()
                    }
                }
            } else {
                let body = resp.text().await.unwrap_or_default();
                state.logs.write().await.add(
                    "error",
                    &format!(
                        "[ERROR] Upstream error HTTP {}: {}",
                        status,
                        safe_truncate(&body, 500)
                    ),
                );
                // 标记 Flow 失败
                if let Some(fid) = &flow_id {
                    let error =
                        FlowError::new(FlowErrorType::from_status_code(status.as_u16()), &body)
                            .with_status_code(status.as_u16());
                    state.flow_monitor.fail_flow(fid, error).await;
                }
                (
                    StatusCode::from_u16(status.as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                    Json(
                        serde_json::json!({"error": {"message": format!("Upstream error: {}", body)}}),
                    ),
                )
                    .into_response()
            }
        }
        Err(e) => {
            // 详细记录网络/连接错误
            let error_details = format!("{e:?}");
            state
                .logs
                .write()
                .await
                .add("error", &format!("[ERROR] Kiro API call failed: {e}"));
            state.logs.write().await.add(
                "debug",
                &format!("[ERROR] Full error details: {error_details}"),
            );
            // 标记 Flow 失败
            if let Some(fid) = &flow_id {
                let error = FlowError::new(FlowErrorType::Network, &e.to_string());
                state.flow_monitor.fail_flow(fid, error).await;
            }
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": {"message": e.to_string()}})),
            )
                .into_response()
        }
    }
}

// ============================================================================
// 流式传输辅助函数
// ============================================================================

/// 获取目标流式格式
///
/// 根据请求路径确定目标流式格式。
///
/// # 参数
/// - `path`: 请求路径
///
/// # 返回
/// 目标流式格式
fn get_target_stream_format(path: &str) -> StreamingFormat {
    if path.contains("/v1/messages") {
        // Anthropic 格式端点
        StreamingFormat::AnthropicSse
    } else {
        // OpenAI 格式端点
        StreamingFormat::OpenAiSse
    }
}

/// 检查是否应该使用真正的流式传输
///
/// 根据凭证类型和配置决定是否使用真正的流式传输。
/// 目前，只有当 Provider 实现了 StreamingProvider trait 时才返回 true。
///
/// # 参数
/// - `credential`: 凭证信息
///
/// # 返回
/// 是否应该使用真正的流式传输
///
/// # 注意
/// 当前所有 Provider 都返回 false，因为 StreamingProvider trait 尚未实现。
/// 一旦任务 6 完成，此函数将根据凭证类型返回适当的值。
fn should_use_true_streaming(
    credential: &crate::models::provider_pool_model::ProviderCredential,
) -> bool {
    use crate::models::provider_pool_model::CredentialData;

    // TODO: 当 StreamingProvider trait 实现后，根据凭证类型返回 true
    // 目前所有 Provider 都使用伪流式模式
    match &credential.credential {
        // Kiro/CodeWhisperer - 需要实现 StreamingProvider
        CredentialData::KiroOAuth { .. } => false,
        // Claude - 需要实现 StreamingProvider
        CredentialData::ClaudeKey { .. } => false,
        // OpenAI - 需要实现 StreamingProvider
        CredentialData::OpenAIKey { .. } => false,
        // Antigravity - 需要实现 StreamingProvider
        CredentialData::AntigravityOAuth { .. } => false,
        // 其他类型暂不支持流式
        _ => false,
    }
}

/// 构建流式错误响应
///
/// 将错误转换为 SSE 格式的错误事件。
///
/// # 参数
/// - `error_type`: 错误类型
/// - `message`: 错误消息
/// - `target_format`: 目标流式格式
///
/// # 返回
/// SSE 格式的错误响应
///
/// # 需求覆盖
/// - 需求 5.3: 流中发生错误时发送错误事件并优雅关闭流
fn build_stream_error_response(
    error_type: &str,
    message: &str,
    target_format: StreamingFormat,
) -> Response {
    let error_event = match target_format {
        StreamingFormat::AnthropicSse => {
            format!(
                "event: error\ndata: {}\n\n",
                serde_json::json!({
                    "type": "error",
                    "error": {
                        "type": error_type,
                        "message": message
                    }
                })
            )
        }
        // TODO: 任务 6 完成后，添加 GeminiStream 分支
        StreamingFormat::OpenAiSse => {
            format!(
                "data: {}\n\n",
                serde_json::json!({
                    "error": {
                        "type": error_type,
                        "message": message
                    }
                })
            )
        }
        StreamingFormat::AwsEventStream => {
            // AWS Event Stream 格式的错误（不太可能作为目标格式）
            format!(
                "data: {}\n\n",
                serde_json::json!({
                    "error": {
                        "type": error_type,
                        "message": message
                    }
                })
            )
        }
    };

    Response::builder()
        .status(StatusCode::OK) // SSE 错误仍然返回 200
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(Body::from(error_event))
        .unwrap_or_else(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": {"message": "Failed to build error response"}})),
            )
                .into_response()
        })
}
