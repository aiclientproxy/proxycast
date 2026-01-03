//! Provider 调用处理器
//!
//! 根据凭证类型调用不同的 Provider API
//!
//! # 流式传输支持

#![allow(dead_code)]
//!
//! 本模块支持真正的端到端流式传输，通过以下组件实现：
//! - `StreamManager`: 管理流式请求的生命周期
//! - `StreamingProvider`: Provider 的流式 API 接口
//! - `FlowMonitor`: 实时捕获流式响应
//! - `handle_kiro_stream()`: Kiro 凭证的真正流式处理（AWS Event Stream → Anthropic SSE）
//!
//! # Kiro 凭证流式处理
//!
//! 当使用 Kiro 凭证且 `stream=true` 时，系统会：
//! 1. 调用 `KiroProvider.call_api_stream()` 获取 AWS Event Stream 格式的流式响应
//! 2. 使用 `AwsEventStreamParser` 实时解析每个 JSON payload
//! 3. 使用 `AnthropicSseGenerator` 转换为 Anthropic SSE 格式
//! 4. 通过 `FlowMonitor.process_chunk()` 记录每个 chunk
//!
//! # 错误处理
//!
//! 流式传输期间的错误处理：
//! - 网络错误：记录日志，发送 SSE 错误事件，调用 FlowMonitor.fail_flow()
//! - 解析错误：记录警告，跳过无效数据，继续处理后续 chunks
//! - 上游错误：将 Provider 返回的错误转发给客户端
//!
//! # 需求覆盖
//!
//! - 需求 1.1: 使用 reqwest 的流式响应模式
//! - 需求 1.2: 实时解析每个 JSON payload 并转换为 Anthropic SSE 事件
//! - 需求 1.3: 立即发送 content_block_delta 事件给客户端
//! - 需求 3.1: Flow Monitor 记录 chunk_count 大于 0
//! - 需求 3.2: 调用 process_chunk 更新流重建器
//! - 需求 4.2: 调用 process_chunk 更新流重建器
//! - 需求 5.1: 流式传输期间发生网络错误时，发出错误事件并以失败状态完成 flow
//! - 需求 5.2: AWS Event Stream 解析失败时记录错误并继续处理后续 chunks
//! - 需求 5.3: 将上游 Provider 返回的错误转发给客户端
//! - 需求 6.1: 流式请求使用 handle_kiro_stream()
//! - 需求 6.2: 非流式请求返回完整 JSON 响应

use axum::{
    body::Body,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures::StreamExt;

use crate::converter::anthropic_to_openai::convert_anthropic_to_openai;
use crate::converter::openai_to_antigravity::{
    convert_antigravity_to_openai_response, convert_openai_to_antigravity_with_context,
};
use crate::flow_monitor::models::{FlowError, FlowErrorType};
use crate::flow_monitor::stream_rebuilder::StreamFormat;
use crate::models::anthropic::AnthropicMessagesRequest;
use crate::models::openai::ChatCompletionRequest;
use crate::models::provider_pool_model::{CredentialData, ProviderCredential};
use crate::providers::{
    AntigravityProvider, ClaudeCustomProvider, KiroProvider, OpenAICustomProvider, VertexProvider,
};
use crate::server::AppState;
use crate::server_utils::{
    build_anthropic_response, build_anthropic_stream_response, parse_cw_response, safe_truncate,
    CWParsedResponse,
};
use crate::stream::{PipelineConfig, StreamPipeline};
use crate::streaming::traits::StreamingProvider;
use crate::streaming::{
    StreamConfig, StreamContext, StreamError, StreamFormat as StreamingFormat, StreamManager,
    StreamResponse,
};

/// 根据凭证调用 Provider (Anthropic 格式)
///
/// # 参数
/// - `state`: 应用状态
/// - `credential`: 凭证信息
/// - `request`: Anthropic 格式请求
/// - `flow_id`: Flow ID（可选，用于流式响应处理）
pub async fn call_provider_anthropic(
    state: &AppState,
    credential: &ProviderCredential,
    request: &AnthropicMessagesRequest,
    flow_id: Option<&str>,
) -> Response {
    // 如果是流式请求且有 flow_id，设置流式状态
    if request.stream {
        if let Some(fid) = flow_id {
            // 根据凭证类型确定流格式
            // 注意：Kiro 凭证虽然原始返回 AWS Event Stream，但 handle_kiro_stream 会将其转换为 Anthropic SSE 格式
            let format = match &credential.credential {
                CredentialData::KiroOAuth { .. } => StreamFormat::Anthropic, // Kiro 流式响应被转换为 Anthropic SSE 格式
                CredentialData::ClaudeKey { .. } => StreamFormat::Anthropic,
                CredentialData::AntigravityOAuth { .. } => StreamFormat::Gemini,
                _ => StreamFormat::Unknown,
            };
            state.flow_monitor.set_streaming(fid, format).await;
        }
    }

    match &credential.credential {
        CredentialData::KiroOAuth { creds_file_path } => {
            // 如果是流式请求，使用真正的流式处理（需求 1.1, 6.1）
            if request.stream {
                return handle_kiro_stream(state, credential, request, flow_id).await;
            }

            // 非流式请求，使用现有的 call_api() 方法（需求 6.1, 6.2, 6.3）
            // 使用 TokenCacheService 获取有效 token
            let db = match &state.db {
                Some(db) => db,
                None => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": "Database not available"}})),
                    )
                        .into_response();
                }
            };
            // 获取缓存的 token
            let token = match state
                .token_cache
                .get_valid_token(db, &credential.uuid)
                .await
            {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!("[POOL] Token cache miss, loading from source: {}", e);
                    // 回退到从源文件加载
                    let mut kiro = KiroProvider::new();
                    if let Err(e) = kiro.load_credentials_from_path(creds_file_path).await {
                        // 记录凭证加载失败
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&format!("Failed to load credentials: {}", e)),
                        );
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": format!("Failed to load Kiro credentials: {}", e)}})),
                        )
                            .into_response();
                    }
                    if let Err(e) = kiro.refresh_token().await {
                        // 记录 Token 刷新失败
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&format!("Token refresh failed: {}", e)),
                        );
                        return (
                            StatusCode::UNAUTHORIZED,
                            Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {}", e)}})),
                        )
                            .into_response();
                    }
                    kiro.credentials.access_token.unwrap_or_default()
                }
            };
            // 使用获取到的 token 创建 KiroProvider
            let mut kiro = KiroProvider::new();
            // 从源文件加载其他配置（region, profile_arn 等）
            // 注意：必须先加载凭证文件，再设置 token，因为 load_credentials_from_path 会覆盖整个 credentials
            let _ = kiro.load_credentials_from_path(creds_file_path).await;
            // 使用缓存的 token 覆盖文件中的 token（缓存的 token 更新）
            kiro.credentials.access_token = Some(token);
            let openai_request = convert_anthropic_to_openai(request);
            let resp = match kiro.call_api(&openai_request).await {
                Ok(r) => r,
                Err(e) => {
                    // 记录 API 调用失败
                    let _ = state.pool_service.mark_unhealthy(
                        db,
                        &credential.uuid,
                        Some(&e.to_string()),
                    );
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                    )
                        .into_response();
                }
            };
            let status = resp.status();
            if status.is_success() {
                match resp.bytes().await {
                    Ok(bytes) => {
                        let body = String::from_utf8_lossy(&bytes).to_string();
                        let parsed = parse_cw_response(&body);
                        // 记录成功
                        let _ = state.pool_service.mark_healthy(
                            db,
                            &credential.uuid,
                            Some(&request.model),
                        );
                        let _ = state.pool_service.record_usage(db, &credential.uuid);
                        // 非流式请求返回完整 JSON 响应（需求 6.2）
                        build_anthropic_response(&request.model, &parsed)
                    }
                    Err(e) => {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": e.to_string()}})),
                        )
                            .into_response()
                    }
                }
            } else if status.as_u16() == 401 || status.as_u16() == 403 {
                // Token 过期，强制刷新并重试
                tracing::info!(
                    "[POOL] Got {}, forcing token refresh for {}",
                    status,
                    &credential.uuid[..8]
                );
                let new_token = match state
                    .token_cache
                    .refresh_and_cache(db, &credential.uuid, true)
                    .await
                {
                    Ok(t) => t,
                    Err(e) => {
                        // 记录 Token 刷新失败
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&format!("Token refresh failed: {}", e)),
                        );
                        return (
                            StatusCode::UNAUTHORIZED,
                            Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {}", e)}})),
                        )
                            .into_response();
                    }
                };
                // 使用新 token 重试
                kiro.credentials.access_token = Some(new_token);
                match kiro.call_api(&openai_request).await {
                    Ok(retry_resp) => {
                        if retry_resp.status().is_success() {
                            match retry_resp.bytes().await {
                                Ok(bytes) => {
                                    let body = String::from_utf8_lossy(&bytes).to_string();
                                    let parsed = parse_cw_response(&body);
                                    // 记录重试成功
                                    let _ = state.pool_service.mark_healthy(
                                        db,
                                        &credential.uuid,
                                        Some(&request.model),
                                    );
                                    let _ = state.pool_service.record_usage(db, &credential.uuid);
                                    // 非流式请求返回完整 JSON 响应（需求 6.2）
                                    build_anthropic_response(&request.model, &parsed)
                                }
                                Err(e) => {
                                    let _ = state.pool_service.mark_unhealthy(
                                        db,
                                        &credential.uuid,
                                        Some(&e.to_string()),
                                    );
                                    (
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                                    )
                                        .into_response()
                                }
                            }
                        } else {
                            let body = retry_resp.text().await.unwrap_or_default();
                            let _ = state.pool_service.mark_unhealthy(
                                db,
                                &credential.uuid,
                                Some(&format!("Retry failed: {}", body)),
                            );
                            (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(serde_json::json!({"error": {"message": format!("Retry failed: {}", body)}})),
                            )
                                .into_response()
                        }
                    }
                    Err(e) => {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": e.to_string()}})),
                        )
                            .into_response()
                    }
                }
            } else {
                let body = resp.text().await.unwrap_or_default();
                let _ = state
                    .pool_service
                    .mark_unhealthy(db, &credential.uuid, Some(&body));
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": body}})),
                )
                    .into_response()
            }
        }
        CredentialData::GeminiOAuth { .. } => {
            // Gemini OAuth 路由暂不支持
            (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({"error": {"message": "Gemini OAuth routing not yet implemented. Use /v1/messages with Gemini models instead."}})),
            )
                .into_response()
        }
        CredentialData::QwenOAuth { .. } => {
            // Qwen OAuth 路由暂不支持
            (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({"error": {"message": "Qwen OAuth routing not yet implemented. Use /v1/messages with Qwen models instead."}})),
            )
                .into_response()
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
                // 记录凭证加载失败
                if let Some(db) = &state.db {
                    let _ = state.pool_service.mark_unhealthy(
                        db,
                        &credential.uuid,
                        Some(&format!("Failed to load credentials: {}", e)),
                    );
                }
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": format!("Failed to load Antigravity credentials: {}", e)}})),
                )
                    .into_response();
            }

            // 使用新的 validate_token() 方法检查 Token 状态
            let validation_result = antigravity.validate_token();
            tracing::info!("[Antigravity] Token 验证结果: {:?}", validation_result);

            // 根据验证结果决定是否刷新
            if validation_result.needs_refresh() {
                tracing::info!("[Antigravity] Token 需要刷新，开始刷新...");
                match antigravity.refresh_token_with_retry(3).await {
                    Ok(new_token) => {
                        tracing::info!("[Antigravity] Token 刷新成功，新 token 长度: {}", new_token.len());
                        // 刷新成功，标记为健康
                        if let Some(db) = &state.db {
                            let _ = state.pool_service.mark_healthy(
                                db,
                                &credential.uuid,
                                None,
                            );
                        }
                    }
                    Err(refresh_error) => {
                        tracing::error!("[Antigravity] Token 刷新失败: {:?}", refresh_error);
                        // 使用新的 mark_unhealthy_with_details 方法
                        if let Some(db) = &state.db {
                            let _ = state.pool_service.mark_unhealthy_with_details(
                                db,
                                &credential.uuid,
                                &refresh_error,
                            );
                        }

                        // 根据错误类型返回不同的状态码和消息
                        let (status, message) = if refresh_error.requires_reauth() {
                            (StatusCode::UNAUTHORIZED, refresh_error.user_message())
                        } else {
                            (StatusCode::INTERNAL_SERVER_ERROR, refresh_error.user_message())
                        };

                        return (
                            status,
                            Json(serde_json::json!({"error": {"message": message}})),
                        )
                            .into_response();
                    }
                }
            }

            // 设置项目 ID
            if let Some(pid) = project_id {
                antigravity.project_id = Some(pid.clone());
            } else if let Err(e) = antigravity.discover_project().await {
                tracing::warn!("[Antigravity] Failed to discover project: {}", e);
            }
            // 获取 project_id 用于请求
            let proj_id = antigravity.project_id.clone().unwrap_or_default();
            // 先转换为 OpenAI 格式，再转换为 Antigravity 格式
            let openai_request = convert_anthropic_to_openai(request);
            let antigravity_request = convert_openai_to_antigravity_with_context(&openai_request, &proj_id);
            match antigravity
                .generate_content(&request.model, &antigravity_request)
                .await
            {
                Ok(resp) => {
                    // 转换为 OpenAI 格式，再构建 Anthropic 响应
                    let content = resp["candidates"][0]["content"]["parts"][0]["text"]
                        .as_str()
                        .unwrap_or("");
                    let parsed = CWParsedResponse {
                        content: content.to_string(),
                        tool_calls: Vec::new(),
                        usage_credits: 0.0,
                        context_usage_percentage: 0.0,
                    };
                    // 记录成功
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_healthy(
                            db,
                            &credential.uuid,
                            Some(&request.model),
                        );
                        let _ = state.pool_service.record_usage(db, &credential.uuid);
                    }
                    if request.stream {
                        build_anthropic_stream_response(&request.model, &parsed)
                    } else {
                        build_anthropic_response(&request.model, &parsed)
                    }
                }
                Err(e) => {
                    // 记录 API 调用失败
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                    }
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                    )
                        .into_response()
                }
            }
        }
        CredentialData::OpenAIKey { api_key, base_url } => {
            let openai = OpenAICustomProvider::with_config(api_key.clone(), base_url.clone());
            let openai_request = convert_anthropic_to_openai(request);
            match openai.call_api(&openai_request).await {
                Ok(resp) => {
                    if resp.status().is_success() {
                        match resp.text().await {
                            Ok(body) => {
                                if let Ok(openai_resp) =
                                    serde_json::from_str::<serde_json::Value>(&body)
                                {
                                    let content = openai_resp["choices"][0]["message"]["content"]
                                        .as_str()
                                        .unwrap_or("");
                                    let parsed = CWParsedResponse {
                                        content: content.to_string(),
                                        tool_calls: Vec::new(),
                                        usage_credits: 0.0,
                                        context_usage_percentage: 0.0,
                                    };
                                    // 记录成功
                                    if let Some(db) = &state.db {
                                        let _ = state.pool_service.mark_healthy(
                                            db,
                                            &credential.uuid,
                                            Some(&request.model),
                                        );
                                        let _ =
                                            state.pool_service.record_usage(db, &credential.uuid);
                                    }
                                    if request.stream {
                                        build_anthropic_stream_response(&request.model, &parsed)
                                    } else {
                                        build_anthropic_response(&request.model, &parsed)
                                    }
                                } else {
                                    // 记录解析失败
                                    if let Some(db) = &state.db {
                                        let _ = state.pool_service.mark_unhealthy(
                                            db,
                                            &credential.uuid,
                                            Some("Failed to parse OpenAI response"),
                                        );
                                    }
                                    (
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        Json(serde_json::json!({"error": {"message": "Failed to parse OpenAI response"}})),
                                    )
                                        .into_response()
                                }
                            }
                            Err(e) => {
                                if let Some(db) = &state.db {
                                    let _ = state.pool_service.mark_unhealthy(
                                        db,
                                        &credential.uuid,
                                        Some(&e.to_string()),
                                    );
                                }
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    Json(serde_json::json!({"error": {"message": e.to_string()}})),
                                )
                                    .into_response()
                            }
                        }
                    } else {
                        let body = resp.text().await.unwrap_or_default();
                        if let Some(db) = &state.db {
                            let _ = state.pool_service.mark_unhealthy(
                                db,
                                &credential.uuid,
                                Some(&body),
                            );
                        }
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": body}})),
                        )
                            .into_response()
                    }
                }
                Err(e) => {
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                    }
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                    )
                        .into_response()
                }
            }
        }
        CredentialData::ClaudeKey { api_key, base_url } => {
            // 打印 Claude 代理 URL 用于调试
            let actual_base_url = base_url.as_deref().unwrap_or("https://api.anthropic.com");
            let claude = ClaudeCustomProvider::with_config(api_key.clone(), base_url.clone());
            let request_url = claude.get_base_url();
            state.logs.write().await.add(
                "info",
                &format!(
                    "[CLAUDE] 使用 Claude API 代理: base_url={} -> {}/v1/messages credential_uuid={} stream={}",
                    actual_base_url,
                    request_url,
                    &credential.uuid[..8],
                    request.stream
                ),
            );
            // 打印请求参数
            let request_json = serde_json::to_string(request).unwrap_or_default();
            state.logs.write().await.add(
                "debug",
                &format!(
                    "[CLAUDE] 请求参数: {}",
                    &request_json.chars().take(500).collect::<String>()
                ),
            );
            match claude.call_api(request).await {
                Ok(resp) => {
                    let status = resp.status();
                    // 打印响应状态
                    state.logs.write().await.add(
                        "info",
                        &format!(
                            "[CLAUDE] 响应状态: status={} model={} stream={}",
                            status,
                            request.model,
                            request.stream
                        ),
                    );

                    // 如果是流式请求，直接透传流式响应
                    if request.stream && status.is_success() {
                        state.logs.write().await.add(
                            "info",
                            "[CLAUDE] 流式请求，透传 SSE 响应",
                        );
                        // 记录成功
                        if let Some(db) = &state.db {
                            let _ = state.pool_service.mark_healthy(
                                db,
                                &credential.uuid,
                                Some(&request.model),
                            );
                            let _ = state.pool_service.record_usage(db, &credential.uuid);
                        }
                        // 透传流式响应，保持 SSE 格式
                        let stream = resp.bytes_stream();
                        return Response::builder()
                            .status(StatusCode::OK)
                            .header(header::CONTENT_TYPE, "text/event-stream")
                            .header(header::CACHE_CONTROL, "no-cache")
                            .header("Connection", "keep-alive")
                            .body(Body::from_stream(stream))
                            .unwrap_or_else(|_| {
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    Json(serde_json::json!({"error": {"message": "Failed to build stream response"}})),
                                )
                                    .into_response()
                            });
                    }

                    // 非流式请求，读取完整响应
                    match resp.text().await {
                        Ok(body) => {
                            if status.is_success() {
                                // 打印响应内容预览
                                state.logs.write().await.add(
                                    "debug",
                                    &format!(
                                        "[CLAUDE] 响应内容: {}",
                                        &body.chars().take(500).collect::<String>()
                                    ),
                                );
                                // 记录成功
                                if let Some(db) = &state.db {
                                    let _ = state.pool_service.mark_healthy(
                                        db,
                                        &credential.uuid,
                                        Some(&request.model),
                                    );
                                    let _ = state.pool_service.record_usage(db, &credential.uuid);
                                }
                                Response::builder()
                                    .status(StatusCode::OK)
                                    .header(header::CONTENT_TYPE, "application/json")
                                    .body(Body::from(body))
                                    .unwrap_or_else(|_| {
                                        (
                                            StatusCode::INTERNAL_SERVER_ERROR,
                                            Json(serde_json::json!({"error": {"message": "Failed to build response"}})),
                                        )
                                            .into_response()
                                    })
                            } else {
                                state.logs.write().await.add(
                                    "error",
                                    &format!(
                                        "[CLAUDE] 请求失败: status={} body={}",
                                        status,
                                        &body.chars().take(200).collect::<String>()
                                    ),
                                );
                                if let Some(db) = &state.db {
                                    let _ = state.pool_service.mark_unhealthy(
                                        db,
                                        &credential.uuid,
                                        Some(&body),
                                    );
                                }
                                (
                                    StatusCode::from_u16(status.as_u16())
                                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                                    Json(serde_json::json!({"error": {"message": body}})),
                                )
                                    .into_response()
                            }
                        }
                        Err(e) => {
                            state.logs.write().await.add(
                                "error",
                                &format!("[CLAUDE] 读取响应失败: {}", e),
                            );
                            if let Some(db) = &state.db {
                                let _ = state.pool_service.mark_unhealthy(
                                    db,
                                    &credential.uuid,
                                    Some(&e.to_string()),
                                );
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
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&e.to_string()),
                        );
                    }
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                    )
                        .into_response()
                }
            }
        }
        CredentialData::VertexKey { api_key, base_url, .. } => {
            // Vertex AI uses Gemini-compatible API, convert Anthropic to OpenAI format first
            let openai_request = convert_anthropic_to_openai(request);
            let vertex = VertexProvider::with_config(api_key.clone(), base_url.clone());
            match vertex.chat_completions(&serde_json::to_value(&openai_request).unwrap_or_default()).await {
                Ok(resp) => {
                    let status = resp.status();
                    match resp.text().await {
                        Ok(body) => {
                            if status.is_success() {
                                if let Some(db) = &state.db {
                                    let _ = state.pool_service.mark_healthy(db, &credential.uuid, Some(&request.model));
                                    let _ = state.pool_service.record_usage(db, &credential.uuid);
                                }
                                Response::builder()
                                    .status(StatusCode::OK)
                                    .header(header::CONTENT_TYPE, "application/json")
                                    .body(Body::from(body))
                                    .unwrap_or_else(|_| {
                                        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": "Failed to build response"}}))).into_response()
                                    })
                            } else {
                                if let Some(db) = &state.db {
                                    let _ = state.pool_service.mark_unhealthy(db, &credential.uuid, Some(&body));
                                }
                                (StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), Json(serde_json::json!({"error": {"message": body}}))).into_response()
                            }
                        }
                        Err(e) => {
                            if let Some(db) = &state.db {
                                let _ = state.pool_service.mark_unhealthy(db, &credential.uuid, Some(&e.to_string()));
                            }
                            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": e.to_string()}}))).into_response()
                        }
                    }
                }
                Err(e) => {
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(db, &credential.uuid, Some(&e.to_string()));
                    }
                    (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": e.to_string()}}))).into_response()
                }
            }
        }
        // Gemini API Key credentials - not supported for Anthropic format
        CredentialData::GeminiApiKey { .. } => {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": {"message": "Gemini API Key credentials do not support Anthropic format"}})),
            )
                .into_response()
        }
        // 新增的凭证类型暂不支持 Anthropic 格式
        CredentialData::CodexOAuth { .. }
        | CredentialData::ClaudeOAuth { .. }
        | CredentialData::IFlowOAuth { .. }
        | CredentialData::IFlowCookie { .. } => {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": {"message": "This credential type does not support Anthropic format yet"}})),
            )
                .into_response()
        }
    }
}

/// 根据凭证调用 Provider (OpenAI 格式)
///
/// # 参数
/// - `state`: 应用状态
/// - `credential`: 凭证信息
/// - `request`: OpenAI 格式请求
/// - `flow_id`: Flow ID（可选，用于流式响应处理）
pub async fn call_provider_openai(
    state: &AppState,
    credential: &ProviderCredential,
    request: &ChatCompletionRequest,
    _flow_id: Option<&str>,
) -> Response {
    let _start_time = std::time::Instant::now();

    // 调试：打印凭证类型
    let cred_type = match &credential.credential {
        CredentialData::KiroOAuth { .. } => "KiroOAuth",
        CredentialData::ClaudeKey { .. } => "ClaudeKey",
        CredentialData::OpenAIKey { .. } => "OpenAIKey",
        CredentialData::GeminiOAuth { .. } => "GeminiOAuth",
        CredentialData::GeminiApiKey { .. } => "GeminiApiKey",
        CredentialData::VertexKey { .. } => "VertexKey",
        CredentialData::QwenOAuth { .. } => "QwenOAuth",
        CredentialData::AntigravityOAuth { .. } => "AntigravityOAuth",
        _ => "Other",
    };
    tracing::info!(
        "[CALL_PROVIDER_OPENAI] 凭证类型={}, 凭证名称={:?}, provider_type={}, uuid={}",
        cred_type,
        credential.name,
        credential.provider_type,
        &credential.uuid[..8]
    );

    match &credential.credential {
        CredentialData::KiroOAuth { creds_file_path } => {
            // 优先使用 token cache，避免每次都刷新 token
            let db = match &state.db {
                Some(db) => db,
                None => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": "Database not available"}})),
                    )
                        .into_response();
                }
            };

            // 获取缓存的 token（自动处理过期和刷新）
            let token = match state
                .token_cache
                .get_valid_token(db, &credential.uuid)
                .await
            {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!("[POOL] Token cache miss, loading from source: {}", e);
                    // 降级：从源文件加载并刷新
                    let mut kiro = KiroProvider::new();
                    if let Err(e) = kiro.load_credentials_from_path(creds_file_path).await {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&format!("Failed to load credentials: {}", e)),
                        );
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": format!("Failed to load Kiro credentials: {}", e)}})),
                        )
                            .into_response();
                    }
                    if let Err(e) = kiro.refresh_token().await {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&format!("Token refresh failed: {}", e)),
                        );
                        return (
                            StatusCode::UNAUTHORIZED,
                            Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {}", e)}})),
                        )
                            .into_response();
                    }
                    kiro.credentials.access_token.unwrap_or_default()
                }
            };

            // 使用获取到的 token 创建 KiroProvider
            let mut kiro = KiroProvider::new();
            // 从源文件加载其他配置（region, profile_arn 等）
            // 注意：必须先加载凭证文件，再设置 token，因为 load_credentials_from_path 会覆盖整个 credentials
            let _ = kiro.load_credentials_from_path(creds_file_path).await;
            // 使用缓存的 token 覆盖文件中的 token（缓存的 token 更新）
            kiro.credentials.access_token = Some(token);

            tracing::info!("[CALL_PROVIDER_OPENAI] request.stream = {}, model = {}", request.stream, request.model);

            // 检查是否为流式请求
            if request.stream {
                // 流式请求处理
                tracing::info!("[OPENAI_STREAM] 处理流式请求, model={}", request.model);
                match kiro.call_api_stream(request).await {
                    Ok(stream_response) => {
                        // 记录成功
                        if let Some(db) = &state.db {
                            let _ = state.pool_service.mark_healthy(db, &credential.uuid, Some(&request.model));
                            let _ = state.pool_service.record_usage(db, &credential.uuid);
                        }

                        tracing::info!("[OPENAI_STREAM] 开始转换流式响应");

                        // 使用新的统一流处理管道 (Kiro → OpenAI)
                        let config = PipelineConfig::kiro_to_openai(request.model.clone());
                        let pipeline = std::sync::Arc::new(tokio::sync::Mutex::new(
                            StreamPipeline::new(config),
                        ));

                        // 创建转换流
                        let pipeline_for_stream = pipeline.clone();
                        let pipeline_for_finalize = pipeline.clone();
                        let final_stream = async_stream::stream! {
                            use futures::StreamExt;

                            let mut stream_response = stream_response;

                            while let Some(chunk_result) = stream_response.next().await {
                                match chunk_result {
                                    Ok(bytes) => {
                                        tracing::debug!(
                                            "[OPENAI_STREAM] 收到 {} 字节数据",
                                            bytes.len()
                                        );

                                        // 使用 Pipeline 处理 chunk
                                        let sse_events = {
                                            let mut pipeline_guard = pipeline_for_stream.lock().await;
                                            pipeline_guard.process_chunk(&bytes)
                                        };

                                        tracing::debug!(
                                            "[OPENAI_STREAM] 生成 {} 个 SSE 事件",
                                            sse_events.len()
                                        );

                                        // yield 每个 SSE 事件
                                        for sse_str in sse_events {
                                            yield Ok::<String, StreamError>(sse_str);
                                        }
                                    }
                                    Err(e) => {
                                        tracing::error!("[OPENAI_STREAM] 流式传输错误: {}", e);
                                        yield Err(e);
                                        return;
                                    }
                                }
                            }

                            tracing::info!("[OPENAI_STREAM] 流结束，生成 finalize 事件");

                            // 流结束，使用 Pipeline 生成结束事件
                            let final_events = {
                                let mut pipeline_guard = pipeline_for_finalize.lock().await;
                                pipeline_guard.finish()
                            };

                            tracing::info!("[OPENAI_STREAM] finalize 生成 {} 个事件", final_events.len());

                            for sse_str in final_events {
                                yield Ok::<String, StreamError>(sse_str);
                            }
                        };

                        tracing::info!("[OPENAI_STREAM] 构建 SSE 响应");

                        // 转换为 Body 流
                        let body_stream = final_stream.map(|result| -> Result<axum::body::Bytes, std::io::Error> {
                            match result {
                                Ok(event) => Ok(axum::body::Bytes::from(event)),
                                Err(e) => Ok(axum::body::Bytes::from(e.to_sse_error())),
                            }
                        });

                        // 构建 SSE 响应
                        return Response::builder()
                            .status(StatusCode::OK)
                            .header(header::CONTENT_TYPE, "text/event-stream")
                            .header(header::CACHE_CONTROL, "no-cache")
                            .header(header::CONNECTION, "keep-alive")
                            .header(header::TRANSFER_ENCODING, "chunked")
                            .header("X-Accel-Buffering", "no")
                            .body(Body::from_stream(body_stream))
                            .unwrap_or_else(|_| {
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    Json(
                                        serde_json::json!({"error": {"message": "Failed to build streaming response"}}),
                                    ),
                                )
                                    .into_response()
                            });
                    }
                    Err(e) => {
                        // 记录请求错误
                        if let Some(db) = &state.db {
                            let _ = state.pool_service.mark_unhealthy(db, &credential.uuid, Some(&e.to_string()));
                        }
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": e.to_string()}})),
                        )
                            .into_response();
                    }
                }
            }

            // 非流式请求处理
            match kiro.call_api(request).await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        // 记录成功
                        if let Some(db) = &state.db {
                            let _ = state.pool_service.mark_healthy(db, &credential.uuid, Some(&request.model));
                            let _ = state.pool_service.record_usage(db, &credential.uuid);
                        }
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
                                        "message": message,
                                        "finish_reason": if has_tool_calls { "tool_calls" } else { "stop" }
                                    }],
                                    "usage": {
                                        "prompt_tokens": 0,
                                        "completion_tokens": 0,
                                        "total_tokens": 0
                                    }
                                }))
                                .into_response()
                            }
                            Err(e) => (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(serde_json::json!({"error": {"message": e.to_string()}})),
                            )
                                .into_response(),
                        }
                    } else {
                        // 记录 API 调用失败
                        let body = resp.text().await.unwrap_or_default();
                        if let Some(db) = &state.db {
                            let _ = state.pool_service.mark_unhealthy(db, &credential.uuid, Some(&format!("HTTP {}: {}", status, safe_truncate(&body, 100))));
                        }
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": body}})),
                        )
                            .into_response()
                    }
                }
                Err(e) => {
                    // 记录请求错误
                    if let Some(db) = &state.db {
                        let _ = state.pool_service.mark_unhealthy(db, &credential.uuid, Some(&e.to_string()));
                    }
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                    )
                        .into_response()
                }
            }
        }
        CredentialData::GeminiOAuth { .. } => {
            (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({"error": {"message": "Gemini OAuth routing not yet implemented."}})),
            )
                .into_response()
        }
        CredentialData::QwenOAuth { .. } => {
            (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({"error": {"message": "Qwen OAuth routing not yet implemented."}})),
            )
                .into_response()
        }
        CredentialData::AntigravityOAuth { creds_file_path, project_id } => {
            eprintln!("\n========== [ANTIGRAVITY] 开始处理 Antigravity 请求 ==========");
            eprintln!("[ANTIGRAVITY] 凭证文件: {}", creds_file_path);
            eprintln!("[ANTIGRAVITY] 项目ID: {:?}", project_id);
            eprintln!("[ANTIGRAVITY] 模型: {}", request.model);
            eprintln!("[ANTIGRAVITY] 流式: {}", request.stream);

            let mut antigravity = AntigravityProvider::new();
            if let Err(e) = antigravity.load_credentials_from_path(creds_file_path).await {
                eprintln!("[ANTIGRAVITY] 加载凭证失败: {}", e);
                // 记录凭证加载失败
                if let Some(db) = &state.db {
                    let _ = state.pool_service.mark_unhealthy(
                        db,
                        &credential.uuid,
                        Some(&format!("Failed to load credentials: {}", e)),
                    );
                }
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": format!("Failed to load Antigravity credentials: {}", e)}})),
                )
                    .into_response();
            }
            eprintln!("[ANTIGRAVITY] 凭证加载成功");

            // 使用新的 validate_token() 方法检查 Token 状态
            let validation_result = antigravity.validate_token();
            eprintln!("[ANTIGRAVITY] Token 验证结果: {:?}", validation_result);
            eprintln!("[ANTIGRAVITY] needs_refresh() = {}", validation_result.needs_refresh());
            tracing::info!("[Antigravity] Token 验证结果: {:?}", validation_result);

            // 根据验证结果决定是否刷新
            if validation_result.needs_refresh() {
                eprintln!("[ANTIGRAVITY] Token 需要刷新，开始刷新...");
                tracing::info!("[Antigravity] Token 需要刷新，开始刷新...");
                match antigravity.refresh_token_with_retry(3).await {
                    Ok(new_token) => {
                        eprintln!("[ANTIGRAVITY] Token 刷新成功，新 token 长度: {}", new_token.len());
                        tracing::info!("[Antigravity] Token 刷新成功，新 token 长度: {}", new_token.len());
                        // 刷新成功，标记为健康
                        if let Some(db) = &state.db {
                            let _ = state.pool_service.mark_healthy(
                                db,
                                &credential.uuid,
                                None,
                            );
                        }
                    }
                    Err(refresh_error) => {
                        eprintln!("[ANTIGRAVITY] Token 刷新失败: {:?}", refresh_error);
                        tracing::error!("[Antigravity] Token 刷新失败: {:?}", refresh_error);
                        // 使用新的 mark_unhealthy_with_details 方法
                        if let Some(db) = &state.db {
                            let _ = state.pool_service.mark_unhealthy_with_details(
                                db,
                                &credential.uuid,
                                &refresh_error,
                            );
                        }

                        // 根据错误类型返回不同的状态码和消息
                        let (status, message) = if refresh_error.requires_reauth() {
                            (StatusCode::UNAUTHORIZED, refresh_error.user_message())
                        } else {
                            (StatusCode::INTERNAL_SERVER_ERROR, refresh_error.user_message())
                        };

                        return (
                            status,
                            Json(serde_json::json!({"error": {"message": message}})),
                        )
                            .into_response();
                    }
                }
            } else {
                eprintln!("[ANTIGRAVITY] Token 不需要刷新，继续使用现有 Token");
            }

            // 设置项目 ID
            if let Some(pid) = project_id {
                antigravity.project_id = Some(pid.clone());
            } else if let Err(e) = antigravity.discover_project().await {
                tracing::warn!("[Antigravity] Failed to discover project: {}", e);
            }

            tracing::info!("[ANTIGRAVITY] request.stream = {}, model = {}, project_id = {:?}",
                request.stream, request.model, antigravity.project_id);

            // 检查是否为流式请求
            if request.stream {
                tracing::info!("[ANTIGRAVITY_STREAM] ========== 开始处理流式请求 ==========");
                tracing::info!("[ANTIGRAVITY_STREAM] model={}, has_token={}",
                    request.model, antigravity.credentials.access_token.is_some());

                // 检查是否是图片生成模型
                // 注意：gemini-3-pro-image-preview 是支持图片理解的模型，不是图片生成模型
                // 只有明确的图片生成模型才需要走非流式路径
                let is_image_generation_model = request.model == "imagen"
                    || request.model.starts_with("imagen-")
                    || request.model.contains("image-generation");
                tracing::info!("[ANTIGRAVITY_STREAM] is_image_generation_model={}", is_image_generation_model);

                // 对于图片生成模型，使用非流式请求然后模拟流式返回
                if is_image_generation_model {
                    tracing::info!("[ANTIGRAVITY_STREAM] 图片生成模型，使用非流式请求");

                    // 获取 project_id 用于请求
                    let proj_id = antigravity.project_id.clone().unwrap_or_default();
                    // 转换请求格式 - 这已经是完整的 Antigravity 请求格式
                    let antigravity_request = convert_openai_to_antigravity_with_context(request, &proj_id);

                    // 直接调用 call_api，因为 antigravity_request 已经是完整格式
                    match antigravity.call_api("generateContent", &antigravity_request).await {
                        Ok(resp) => {
                            // 保存原始响应到文件用于调试
                            let resp_str = serde_json::to_string_pretty(&resp).unwrap_or_default();
                            let debug_dir = dirs::home_dir()
                                .map(|h| h.join(".proxycast/logs"))
                                .unwrap_or_else(|| std::path::PathBuf::from("/tmp"));
                            let _ = std::fs::create_dir_all(&debug_dir);
                            let debug_file = debug_dir.join("antigravity_image_response.json");
                            let _ = std::fs::write(&debug_file, &resp_str);
                            tracing::info!("[ANTIGRAVITY_STREAM] 原始响应已保存到: {:?}, 大小: {} bytes", debug_file, resp_str.len());
                            eprintln!("[ANTIGRAVITY_STREAM] 原始响应已保存到: {:?}, 大小: {} bytes", debug_file, resp_str.len());

                            tracing::info!("[ANTIGRAVITY_STREAM] 图片生成完成，转换为流式响应");

                            // 将非流式响应转换为 OpenAI 格式
                            let openai_response = convert_antigravity_to_openai_response(&resp, &request.model);

                            // 保存转换后的响应到文件
                            let openai_str = serde_json::to_string_pretty(&openai_response).unwrap_or_default();
                            let openai_debug_file = debug_dir.join("antigravity_image_openai_response.json");
                            let _ = std::fs::write(&openai_debug_file, &openai_str);
                            tracing::info!("[ANTIGRAVITY_STREAM] OpenAI 响应已保存到: {:?}, 大小: {} bytes", openai_debug_file, openai_str.len());
                            eprintln!("[ANTIGRAVITY_STREAM] OpenAI 响应已保存到: {:?}, 大小: {} bytes", openai_debug_file, openai_str.len());

                            // 将非流式响应转换为流式 SSE 格式
                            let model = request.model.clone();
                            let chunk_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
                            let created = chrono::Utc::now().timestamp();

                            // 提取内容
                            let content = openai_response
                                .get("choices")
                                .and_then(|c| c.as_array())
                                .and_then(|arr| arr.first())
                                .and_then(|choice| choice.get("message"))
                                .and_then(|msg| msg.get("content"))
                                .and_then(|c| c.as_str())
                                .unwrap_or("");

                            tracing::info!("[ANTIGRAVITY_STREAM] 图片内容长度: {} 字符", content.len());
                            eprintln!("[ANTIGRAVITY_STREAM] 图片内容长度: {} 字符", content.len());

                            // 构建 SSE 事件
                            let mut sse_events = String::new();

                            // 发送内容 chunk
                            if !content.is_empty() {
                                let chunk_response = serde_json::json!({
                                    "id": chunk_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {
                                            "content": content
                                        },
                                        "finish_reason": null
                                    }]
                                });
                                sse_events.push_str(&format!("data: {}\n\n", chunk_response.to_string()));
                            }

                            // 发送结束 chunk
                            let done_response = serde_json::json!({
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }]
                            });
                            sse_events.push_str(&format!("data: {}\n\n", done_response.to_string()));
                            sse_events.push_str("data: [DONE]\n\n");

                            return Response::builder()
                                .status(StatusCode::OK)
                                .header(header::CONTENT_TYPE, "text/event-stream")
                                .header(header::CACHE_CONTROL, "no-cache")
                                .header(header::CONNECTION, "keep-alive")
                                .body(Body::from(sse_events))
                                .unwrap_or_else(|_| {
                                    (
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        Json(serde_json::json!({"error": {"message": "Failed to build streaming response"}})),
                                    )
                                        .into_response()
                                });
                        }
                        Err(e) => {
                            tracing::error!("[ANTIGRAVITY_STREAM] 图片生成失败: {}", e);
                            return (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(serde_json::json!({"error": {"message": e.to_string()}})),
                            )
                                .into_response();
                        }
                    }
                }

                match antigravity.call_api_stream(request).await {
                    Ok(stream_response) => {
                        eprintln!("[ANTIGRAVITY_STREAM] ✓ 流式响应已建立");
                        tracing::info!("[ANTIGRAVITY_STREAM] ✓ 流式响应已建立");

                        let model = request.model.clone();

                        // Antigravity 返回的是分片的 JSON，需要累积所有数据后解析
                        // 使用 channel 来收集所有数据，然后一次性返回
                        let (tx, rx) = tokio::sync::oneshot::channel::<Result<String, String>>();

                        // 在后台任务中收集所有数据
                        let model_clone = model.clone();
                        tokio::spawn(async move {
                            use futures::StreamExt;
                            let mut stream = stream_response;
                            let mut all_data = String::new();
                            let mut chunk_count = 0u32;

                            while let Some(result) = stream.next().await {
                                chunk_count += 1;
                                match result {
                                    Ok(bytes) => {
                                        let text = String::from_utf8_lossy(&bytes);
                                        all_data.push_str(&text);

                                        if chunk_count <= 3 {
                                            eprintln!("[ANTIGRAVITY_STREAM] 收集 chunk #{}: {} bytes", chunk_count, bytes.len());
                                        } else if chunk_count % 200 == 0 {
                                            eprintln!("[ANTIGRAVITY_STREAM] 已收集 {} 个 chunk, 总大小: {} bytes", chunk_count, all_data.len());
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("[ANTIGRAVITY_STREAM] chunk #{} 错误: {}", chunk_count, e);
                                        let _ = tx.send(Err(e.to_string()));
                                        return;
                                    }
                                }
                            }

                            eprintln!("[ANTIGRAVITY_STREAM] 流结束，共收集 {} 个 chunk, 总大小: {} bytes", chunk_count, all_data.len());

                            // 尝试解析累积的 JSON 数据
                            // Antigravity 返回格式: { "response": { "candidates": [...] } }
                            let result = parse_antigravity_accumulated_response(&all_data, &model_clone);
                            let _ = tx.send(result);
                        });

                        // 等待数据收集完成，然后构建 SSE 响应
                        let sse_stream = async_stream::stream! {
                            match rx.await {
                                Ok(Ok(sse_content)) => {
                                    // 返回累积的 SSE 事件
                                    yield Ok::<_, std::io::Error>(axum::body::Bytes::from(sse_content));
                                }
                                Ok(Err(e)) => {
                                    eprintln!("[ANTIGRAVITY_STREAM] 解析错误: {}", e);
                                    let error_event = format!(
                                        "data: {{\"error\": {{\"message\": \"{}\"}}}}\n\ndata: [DONE]\n\n",
                                        e.replace("\"", "\\\"")
                                    );
                                    yield Ok(axum::body::Bytes::from(error_event));
                                }
                                Err(_) => {
                                    eprintln!("[ANTIGRAVITY_STREAM] channel 接收错误");
                                    let error_event = "data: {\"error\": {\"message\": \"Internal error\"}}\n\ndata: [DONE]\n\n";
                                    yield Ok(axum::body::Bytes::from(error_event.to_string()));
                                }
                            }
                        };

                        return Response::builder()
                            .status(StatusCode::OK)
                            .header(header::CONTENT_TYPE, "text/event-stream")
                            .header(header::CACHE_CONTROL, "no-cache")
                            .header(header::CONNECTION, "keep-alive")
                            .header("X-Accel-Buffering", "no")
                            .body(Body::from_stream(sse_stream))
                            .unwrap_or_else(|_| {
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    Json(
                                        serde_json::json!({"error": {"message": "Failed to build streaming response"}}),
                                    ),
                                )
                                    .into_response()
                            });
                    }
                    Err(e) => {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": e.to_string()}})),
                        )
                            .into_response();
                    }
                }
            }

            // 非流式请求处理
            // 获取 project_id 用于请求
            let proj_id = antigravity.project_id.clone().unwrap_or_default();
            // 转换请求格式
            let antigravity_request = convert_openai_to_antigravity_with_context(request, &proj_id);
            match antigravity.generate_content(&request.model, &antigravity_request).await {
                Ok(resp) => {
                    let openai_response = convert_antigravity_to_openai_response(&resp, &request.model);
                    Json(openai_response).into_response()
                }
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": e.to_string()}})),
                )
                    .into_response(),
            }
        }
        CredentialData::OpenAIKey { api_key, base_url } => {
            let openai = OpenAICustomProvider::with_config(api_key.clone(), base_url.clone());

            tracing::info!("[OPENAI_KEY] request.stream = {}, model = {}", request.stream, request.model);

            // 检查是否为流式请求
            if request.stream {
                tracing::info!("[OPENAI_KEY_STREAM] 处理流式请求, model={}", request.model);
                match openai.call_api_stream(request).await {
                    Ok(stream_response) => {
                        tracing::info!("[OPENAI_KEY_STREAM] 开始直接转发 OpenAI SSE 流");

                        // OpenAI 提供商已经返回 OpenAI SSE 格式，直接转发
                        let body_stream = stream_response.map(|result| -> Result<axum::body::Bytes, std::io::Error> {
                            match result {
                                Ok(bytes) => Ok(bytes),
                                Err(e) => Ok(axum::body::Bytes::from(e.to_sse_error())),
                            }
                        });

                        return Response::builder()
                            .status(StatusCode::OK)
                            .header(header::CONTENT_TYPE, "text/event-stream")
                            .header(header::CACHE_CONTROL, "no-cache")
                            .header(header::CONNECTION, "keep-alive")
                            .header(header::TRANSFER_ENCODING, "chunked")
                            .header("X-Accel-Buffering", "no")
                            .body(Body::from_stream(body_stream))
                            .unwrap_or_else(|_| {
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    Json(
                                        serde_json::json!({"error": {"message": "Failed to build streaming response"}}),
                                    ),
                                )
                                    .into_response()
                            });
                    }
                    Err(e) => {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": e.to_string()}})),
                        )
                            .into_response();
                    }
                }
            }

            // 非流式请求处理
            match openai.call_api(request).await {
                Ok(resp) => {
                    if resp.status().is_success() {
                        match resp.text().await {
                            Ok(body) => {
                                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                                    Json(json).into_response()
                                } else {
                                    (
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        Json(serde_json::json!({"error": {"message": "Invalid JSON response"}})),
                                    )
                                        .into_response()
                                }
                            }
                            Err(e) => (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(serde_json::json!({"error": {"message": e.to_string()}})),
                            )
                                .into_response(),
                        }
                    } else {
                        let body = resp.text().await.unwrap_or_default();
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": body}})),
                        )
                            .into_response()
                    }
                }
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": e.to_string()}})),
                )
                    .into_response(),
            }
        }
        CredentialData::ClaudeKey { api_key, base_url } => {
            // 打印 Claude 代理 URL 用于调试
            let actual_base_url = base_url.as_deref().unwrap_or("https://api.anthropic.com");
            tracing::info!(
                "[CLAUDE] 使用 Claude API 代理: base_url={} credential_uuid={} stream={}",
                actual_base_url,
                &credential.uuid[..8],
                request.stream
            );
            let claude = ClaudeCustomProvider::with_config(api_key.clone(), base_url.clone());

            // 检查是否为流式请求
            if request.stream {
                tracing::info!("[CLAUDE_KEY_STREAM] 处理流式请求, model={}", request.model);

                match claude.call_api_stream(request).await {
                    Ok(stream_response) => {
                        tracing::info!("[CLAUDE_KEY_STREAM] 开始转换 Anthropic SSE 到 OpenAI SSE");

                        // 创建 StreamConverter 将 Anthropic SSE 转换为 OpenAI SSE
                        let converter = std::sync::Arc::new(tokio::sync::Mutex::new(
                            crate::streaming::converter::StreamConverter::with_model(
                                crate::streaming::converter::StreamFormat::AnthropicSse,
                                crate::streaming::converter::StreamFormat::OpenAiSse,
                                &request.model,
                            ),
                        ));

                        let converter_for_stream = converter.clone();
                        let final_stream = async_stream::stream! {
                            use futures::StreamExt;

                            let mut stream_response = stream_response;

                            while let Some(chunk_result) = stream_response.next().await {
                                match chunk_result {
                                    Ok(bytes) => {
                                        // 转换 Anthropic SSE 到 OpenAI SSE
                                        let sse_events = {
                                            let mut converter_guard = converter_for_stream.lock().await;
                                            converter_guard.convert(&bytes)
                                        };

                                        for sse_str in sse_events {
                                            yield Ok::<String, crate::streaming::StreamError>(sse_str);
                                        }
                                    }
                                    Err(e) => {
                                        tracing::error!("[CLAUDE_KEY_STREAM] 流式传输错误: {}", e);
                                        yield Err(e);
                                        return;
                                    }
                                }
                            }

                            // 流结束，生成结束事件
                            let final_events = {
                                let mut converter_guard = converter_for_stream.lock().await;
                                converter_guard.finish()
                            };

                            for sse_str in final_events {
                                yield Ok::<String, crate::streaming::StreamError>(sse_str);
                            }
                        };

                        let body_stream = final_stream.map(|result| -> Result<axum::body::Bytes, std::io::Error> {
                            match result {
                                Ok(event) => Ok(axum::body::Bytes::from(event)),
                                Err(e) => Ok(axum::body::Bytes::from(e.to_sse_error())),
                            }
                        });

                        return Response::builder()
                            .status(StatusCode::OK)
                            .header(header::CONTENT_TYPE, "text/event-stream")
                            .header(header::CACHE_CONTROL, "no-cache")
                            .header(header::CONNECTION, "keep-alive")
                            .header(header::TRANSFER_ENCODING, "chunked")
                            .header("X-Accel-Buffering", "no")
                            .body(Body::from_stream(body_stream))
                            .unwrap_or_else(|_| {
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    Json(
                                        serde_json::json!({"error": {"message": "Failed to build streaming response"}}),
                                    ),
                                )
                                    .into_response()
                            });
                    }
                    Err(e) => {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": {"message": e.to_string()}})),
                        )
                            .into_response();
                    }
                }
            }

            // 非流式请求处理
            match claude.call_openai_api(request).await {
                Ok(resp) => Json(resp).into_response(),
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": e.to_string()}})),
                )
                    .into_response(),
            }
        }
        CredentialData::VertexKey { api_key, base_url, model_aliases } => {
            // Resolve model alias if present
            let resolved_model = model_aliases.get(&request.model).cloned().unwrap_or_else(|| request.model.clone());
            let mut modified_request = request.clone();
            modified_request.model = resolved_model;
            let vertex = VertexProvider::with_config(api_key.clone(), base_url.clone());
            match vertex.chat_completions(&serde_json::to_value(&modified_request).unwrap_or_default()).await {
                Ok(resp) => {
                    if resp.status().is_success() {
                        match resp.text().await {
                            Ok(body) => {
                                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                                    Json(json).into_response()
                                } else {
                                    (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": "Invalid JSON response"}}))).into_response()
                                }
                            }
                            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": e.to_string()}}))).into_response(),
                        }
                    } else {
                        let body = resp.text().await.unwrap_or_default();
                        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": body}}))).into_response()
                    }
                }
                Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": {"message": e.to_string()}}))).into_response(),
            }
        }
        // Gemini API Key credentials - not supported for OpenAI format yet
        CredentialData::GeminiApiKey { .. } => {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": {"message": "Gemini API Key credentials do not support OpenAI format yet"}})),
            )
                .into_response()
        }
        // 新增的凭证类型暂不支持 OpenAI 格式
        CredentialData::CodexOAuth { .. }
        | CredentialData::ClaudeOAuth { .. }
        | CredentialData::IFlowOAuth { .. }
        | CredentialData::IFlowCookie { .. } => {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": {"message": "This credential type does not support OpenAI format yet"}})),
            )
                .into_response()
        }
    }
}

// ============================================================================
// 流式传输支持
// ============================================================================

/// 获取凭证对应的流式格式
///
/// 根据凭证类型返回对应的流式响应格式。
///
/// # 参数
/// - `credential`: 凭证信息
///
/// # 返回
/// 流式格式枚举
pub fn get_stream_format_for_credential(credential: &ProviderCredential) -> StreamingFormat {
    match &credential.credential {
        CredentialData::KiroOAuth { .. } => StreamingFormat::AwsEventStream,
        CredentialData::ClaudeKey { .. } => StreamingFormat::AnthropicSse,
        CredentialData::OpenAIKey { .. } => StreamingFormat::OpenAiSse,
        // TODO: 任务 6 完成后，将这些改为 GeminiStream
        CredentialData::AntigravityOAuth { .. } => StreamingFormat::OpenAiSse,
        CredentialData::GeminiOAuth { .. } => StreamingFormat::OpenAiSse,
        CredentialData::GeminiApiKey { .. } => StreamingFormat::OpenAiSse,
        CredentialData::VertexKey { .. } => StreamingFormat::OpenAiSse,
        _ => StreamingFormat::OpenAiSse,
    }
}

/// 处理流式响应
///
/// 使用 StreamManager 处理流式响应，集成 Flow Monitor。
///
/// # 参数
/// - `state`: 应用状态
/// - `flow_id`: Flow ID（用于 Flow Monitor 集成）
/// - `source_stream`: 源字节流
/// - `source_format`: 源流格式
/// - `target_format`: 目标流格式
/// - `model`: 模型名称
///
/// # 返回
/// SSE 格式的 HTTP 响应
///
/// # 需求覆盖
/// - 需求 4.2: 调用 process_chunk 更新流重建器
/// - 需求 5.1: 在收到 chunk 后立即转发给客户端
pub async fn handle_streaming_response(
    state: &AppState,
    flow_id: Option<&str>,
    source_stream: StreamResponse,
    source_format: StreamingFormat,
    target_format: StreamingFormat,
    model: &str,
) -> Response {
    // 创建流式管理器
    let manager = StreamManager::with_default_config();

    // 创建流式上下文
    let context = StreamContext::new(
        flow_id.map(|s| s.to_string()),
        source_format,
        target_format,
        model,
    );

    // 获取 flow_id 的克隆用于回调
    let flow_id_for_callback = flow_id.map(|s| s.to_string());
    let flow_monitor = state.flow_monitor.clone();

    // 创建带回调的流式处理
    let managed_stream = if let Some(fid) = flow_id_for_callback {
        // 使用带回调的流式处理，集成 Flow Monitor
        let on_chunk = move |event: &str, _metrics: &crate::streaming::StreamMetrics| {
            // 解析 SSE 事件并调用 process_chunk
            // SSE 格式: "event: xxx\ndata: {...}\n\n"
            let lines: Vec<&str> = event.lines().collect();
            let mut event_type: Option<&str> = None;
            let mut data: Option<&str> = None;

            for line in lines {
                if line.starts_with("event: ") {
                    event_type = Some(&line[7..]);
                } else if line.starts_with("data: ") {
                    data = Some(&line[6..]);
                }
            }

            if let Some(d) = data {
                // 使用 tokio::spawn 异步调用 process_chunk
                let flow_monitor_clone = flow_monitor.clone();
                let fid_clone = fid.clone();
                let event_type_owned = event_type.map(|s| s.to_string());
                let data_owned = d.to_string();

                tokio::spawn(async move {
                    flow_monitor_clone
                        .process_chunk(&fid_clone, event_type_owned.as_deref(), &data_owned)
                        .await;
                });
            }
        };

        let stream = manager.handle_stream_with_callback(context, source_stream, on_chunk);

        // 转换为 Body 流
        let body_stream = stream.map(|result| -> Result<axum::body::Bytes, std::io::Error> {
            match result {
                Ok(event) => Ok(axum::body::Bytes::from(event)),
                Err(e) => Ok(axum::body::Bytes::from(e.to_sse_error())),
            }
        });

        Body::from_stream(body_stream)
    } else {
        // 没有 flow_id，使用普通流式处理
        let stream = manager.handle_stream(context, source_stream);

        let body_stream = stream.map(|result| -> Result<axum::body::Bytes, std::io::Error> {
            match result {
                Ok(event) => Ok(axum::body::Bytes::from(event)),
                Err(e) => Ok(axum::body::Bytes::from(e.to_sse_error())),
            }
        });

        Body::from_stream(body_stream)
    };

    // 构建 SSE 响应
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .header("X-Accel-Buffering", "no")
        .body(managed_stream)
        .unwrap_or_else(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(
                    serde_json::json!({"error": {"message": "Failed to build streaming response"}}),
                ),
            )
                .into_response()
        })
}

/// 处理流式响应（带超时）
///
/// 与 `handle_streaming_response` 类似，但添加了超时保护。
///
/// # 参数
/// - `state`: 应用状态
/// - `flow_id`: Flow ID
/// - `source_stream`: 源字节流
/// - `source_format`: 源流格式
/// - `target_format`: 目标流格式
/// - `model`: 模型名称
/// - `timeout_ms`: 超时时间（毫秒）
///
/// # 返回
/// SSE 格式的 HTTP 响应
///
/// # 需求覆盖
/// - 需求 6.2: 超时错误处理
/// - 需求 6.5: 可配置的流式响应超时
pub async fn handle_streaming_response_with_timeout(
    state: &AppState,
    flow_id: Option<&str>,
    source_stream: StreamResponse,
    source_format: StreamingFormat,
    target_format: StreamingFormat,
    model: &str,
    timeout_ms: u64,
) -> Response {
    use futures::stream::BoxStream;

    // 创建带超时配置的流式管理器
    let config = StreamConfig::new()
        .with_timeout_ms(timeout_ms)
        .with_chunk_timeout_ms(30_000); // 30 秒 chunk 超时

    let manager = StreamManager::new(config.clone());

    // 创建流式上下文
    let context = StreamContext::new(
        flow_id.map(|s| s.to_string()),
        source_format,
        target_format,
        model,
    );

    // 获取 flow_id 的克隆用于回调
    let flow_id_for_callback = flow_id.map(|s| s.to_string());
    let flow_monitor = state.flow_monitor.clone();

    // 创建带超时的流式处理，使用 BoxStream 统一类型
    let timeout_stream: BoxStream<'static, Result<String, crate::streaming::StreamError>> =
        if let Some(fid) = flow_id_for_callback {
            let on_chunk = move |event: &str, _metrics: &crate::streaming::StreamMetrics| {
                let lines: Vec<&str> = event.lines().collect();
                let mut event_type: Option<&str> = None;
                let mut data: Option<&str> = None;

                for line in lines {
                    if line.starts_with("event: ") {
                        event_type = Some(&line[7..]);
                    } else if line.starts_with("data: ") {
                        data = Some(&line[6..]);
                    }
                }

                if let Some(d) = data {
                    let flow_monitor_clone = flow_monitor.clone();
                    let fid_clone = fid.clone();
                    let event_type_owned = event_type.map(|s| s.to_string());
                    let data_owned = d.to_string();

                    tokio::spawn(async move {
                        flow_monitor_clone
                            .process_chunk(&fid_clone, event_type_owned.as_deref(), &data_owned)
                            .await;
                    });
                }
            };

            let stream = manager.handle_stream_with_callback(context, source_stream, on_chunk);
            Box::pin(crate::streaming::with_timeout(stream, &config))
        } else {
            let stream = manager.handle_stream(context, source_stream);
            Box::pin(crate::streaming::with_timeout(stream, &config))
        };

    // 转换为 Body 流
    let body_stream = timeout_stream.map(|result| -> Result<axum::body::Bytes, std::io::Error> {
        match result {
            Ok(event) => Ok(axum::body::Bytes::from(event)),
            Err(e) => Ok(axum::body::Bytes::from(e.to_sse_error())),
        }
    });

    // 构建 SSE 响应
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .header("X-Accel-Buffering", "no")
        .body(Body::from_stream(body_stream))
        .unwrap_or_else(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(
                    serde_json::json!({"error": {"message": "Failed to build streaming response"}}),
                ),
            )
                .into_response()
        })
}

/// 将 reqwest 响应转换为 StreamResponse
///
/// 用于将 Provider 的 HTTP 响应转换为统一的流式响应类型。
///
/// # 参数
/// - `response`: reqwest HTTP 响应
///
/// # 返回
/// 统一的流式响应类型
pub fn response_to_stream(response: reqwest::Response) -> StreamResponse {
    crate::streaming::reqwest_stream_to_stream_response(response)
}

// ============================================================================
// 客户端断开检测
// ============================================================================

/// 带客户端断开检测的流式响应处理
///
/// 在流式传输过程中检测客户端是否断开连接，并在断开时：
/// 1. 停止处理上游数据
/// 2. 标记 Flow 为取消状态
/// 3. 清理资源
///
/// # 参数
/// - `state`: 应用状态
/// - `flow_id`: Flow ID
/// - `source_stream`: 源字节流
/// - `source_format`: 源流格式
/// - `target_format`: 目标流格式
/// - `model`: 模型名称
/// - `cancel_token`: 取消令牌（用于取消上游请求）
///
/// # 返回
/// SSE 格式的 HTTP 响应
///
/// # 需求覆盖
/// - 需求 5.4: 客户端断开时取消上游请求
pub async fn handle_streaming_with_disconnect_detection(
    state: &AppState,
    flow_id: Option<&str>,
    source_stream: StreamResponse,
    source_format: StreamingFormat,
    target_format: StreamingFormat,
    model: &str,
    cancel_token: Option<tokio_util::sync::CancellationToken>,
) -> Response {
    use futures::StreamExt;

    // 创建流式管理器
    let manager = StreamManager::with_default_config();

    // 创建流式上下文
    let context = StreamContext::new(
        flow_id.map(|s| s.to_string()),
        source_format,
        target_format,
        model,
    );

    // 获取 flow_id 的克隆
    let flow_id_for_callback = flow_id.map(|s| s.to_string());
    let flow_id_for_cancel = flow_id.map(|s| s.to_string());
    let flow_monitor = state.flow_monitor.clone();
    let flow_monitor_for_cancel = state.flow_monitor.clone();

    // 创建带回调的流式处理
    // 使用 BoxStream 统一类型
    let managed_stream: futures::stream::BoxStream<
        'static,
        Result<String, crate::streaming::StreamError>,
    > = if let Some(fid) = flow_id_for_callback {
        let on_chunk = move |event: &str, _metrics: &crate::streaming::StreamMetrics| {
            let lines: Vec<&str> = event.lines().collect();
            let mut event_type: Option<&str> = None;
            let mut data: Option<&str> = None;

            for line in lines {
                if line.starts_with("event: ") {
                    event_type = Some(&line[7..]);
                } else if line.starts_with("data: ") {
                    data = Some(&line[6..]);
                }
            }

            if let Some(d) = data {
                let flow_monitor_clone = flow_monitor.clone();
                let fid_clone = fid.clone();
                let event_type_owned = event_type.map(|s| s.to_string());
                let data_owned = d.to_string();

                tokio::spawn(async move {
                    flow_monitor_clone
                        .process_chunk(&fid_clone, event_type_owned.as_deref(), &data_owned)
                        .await;
                });
            }
        };

        Box::pin(manager.handle_stream_with_callback(context, source_stream, on_chunk))
    } else {
        // 没有 flow_id，使用普通流式处理
        Box::pin(manager.handle_stream(context, source_stream))
    };

    // 如果有取消令牌，创建一个可取消的流
    let body_stream = if let Some(token) = cancel_token {
        // 创建一个可取消的流
        let cancellable_stream = CancellableStream::new(managed_stream, token.clone());

        // 当流被取消时，标记 Flow 为取消状态
        let cancel_handler = {
            let token = token.clone();
            let flow_id = flow_id_for_cancel.clone();
            async move {
                token.cancelled().await;
                if let Some(fid) = flow_id {
                    flow_monitor_for_cancel.cancel_flow(&fid).await;
                    tracing::info!("[STREAM] 客户端断开，已取消 Flow: {}", fid);
                }
            }
        };

        // 在后台运行取消处理器
        tokio::spawn(cancel_handler);

        // 转换为 Body 流
        let stream =
            cancellable_stream.map(|result| -> Result<axum::body::Bytes, std::io::Error> {
                match result {
                    Ok(event) => Ok(axum::body::Bytes::from(event)),
                    Err(e) => Ok(axum::body::Bytes::from(e.to_sse_error())),
                }
            });

        Body::from_stream(stream)
    } else {
        // 没有取消令牌，使用普通流
        let stream = managed_stream.map(|result| -> Result<axum::body::Bytes, std::io::Error> {
            match result {
                Ok(event) => Ok(axum::body::Bytes::from(event)),
                Err(e) => Ok(axum::body::Bytes::from(e.to_sse_error())),
            }
        });

        Body::from_stream(stream)
    };

    // 构建 SSE 响应
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .header("X-Accel-Buffering", "no")
        .body(body_stream)
        .unwrap_or_else(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(
                    serde_json::json!({"error": {"message": "Failed to build streaming response"}}),
                ),
            )
                .into_response()
        })
}

/// 可取消的流包装器
///
/// 包装一个流，使其可以通过取消令牌取消。
/// 当取消令牌被触发时，流将返回 ClientDisconnected 错误。
pub struct CancellableStream<S> {
    inner: S,
    cancel_token: tokio_util::sync::CancellationToken,
    cancelled: bool,
}

impl<S> CancellableStream<S> {
    /// 创建新的可取消流
    pub fn new(inner: S, cancel_token: tokio_util::sync::CancellationToken) -> Self {
        Self {
            inner,
            cancel_token,
            cancelled: false,
        }
    }
}

impl<S> futures::Stream for CancellableStream<S>
where
    S: futures::Stream<Item = Result<String, StreamError>> + Unpin,
{
    type Item = Result<String, StreamError>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        use std::task::Poll;

        // 检查是否已取消
        if self.cancelled {
            return Poll::Ready(None);
        }

        // 检查取消令牌
        if self.cancel_token.is_cancelled() {
            self.cancelled = true;
            return Poll::Ready(Some(Err(StreamError::ClientDisconnected)));
        }

        // 轮询内部流
        std::pin::Pin::new(&mut self.inner).poll_next(cx)
    }
}

/// 创建取消令牌
///
/// 创建一个可用于取消流式请求的令牌。
///
/// # 返回
/// 取消令牌
pub fn create_cancel_token() -> tokio_util::sync::CancellationToken {
    tokio_util::sync::CancellationToken::new()
}

/// 检测客户端断开并触发取消
///
/// 监控客户端连接状态，当检测到断开时触发取消令牌。
///
/// # 参数
/// - `cancel_token`: 取消令牌
///
/// # 注意
/// 此函数应该在单独的任务中运行，与流式响应并行。
/// 实际的断开检测依赖于 axum 的连接管理。
pub async fn monitor_client_disconnect(cancel_token: tokio_util::sync::CancellationToken) {
    // 在实际应用中，这里会监控客户端连接状态
    // 当检测到断开时，调用 cancel_token.cancel()
    //
    // 由于 axum 的 SSE 响应会自动处理客户端断开，
    // 这个函数主要用于需要主动检测断开的场景

    // 等待取消令牌被触发（由其他地方触发）
    cancel_token.cancelled().await;
}

// ============================================================================
// Kiro 凭证真正流式响应处理
// ============================================================================

/// Kiro 凭证流式响应处理
///
/// 实现真正的端到端流式传输，将 AWS Event Stream 格式转换为 Anthropic SSE 格式。
///
/// # 参数
/// - `state`: 应用状态
/// - `credential`: Kiro 凭证信息
/// - `request`: Anthropic 格式请求
/// - `flow_id`: Flow ID（可选，用于流式响应处理）
///
/// # 需求覆盖
/// - 需求 1.1: 使用 reqwest 的流式响应模式
/// - 需求 1.2: 实时解析每个 JSON payload 并转换为 Anthropic SSE 事件
/// - 需求 1.3: 立即发送 content_block_delta 事件给客户端
/// - 需求 3.1: Flow Monitor 记录 chunk_count 大于 0
/// - 需求 3.2: 调用 process_chunk 更新流重建器
/// - 需求 3.3: 流完成时拥有完整的重建响应内容
/// - 需求 4.4: 在流式请求前检查 Token 是否即将过期（10分钟内）并提前刷新
pub async fn handle_kiro_stream(
    state: &AppState,
    credential: &ProviderCredential,
    request: &AnthropicMessagesRequest,
    flow_id: Option<&str>,
) -> Response {
    tracing::info!(
        "[KIRO_STREAM] handle_kiro_stream 被调用, model={}, flow_id={:?}",
        request.model,
        flow_id
    );

    // 提取凭证文件路径
    let creds_file_path = match &credential.credential {
        CredentialData::KiroOAuth { creds_file_path } => creds_file_path.clone(),
        _ => {
            tracing::error!("[KIRO_STREAM] 无效的凭证类型");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": {"message": "Invalid credential type for Kiro stream"}})),
            )
                .into_response();
        }
    };

    tracing::info!("[KIRO_STREAM] 凭证文件路径: {}", creds_file_path);

    // 获取数据库连接
    let db = match &state.db {
        Some(db) => db,
        None => {
            tracing::error!("[KIRO_STREAM] 数据库不可用");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": {"message": "Database not available"}})),
            )
                .into_response();
        }
    };

    // 获取有效 token（需求 4.4: 检查 Token 是否即将过期，10分钟内则提前刷新）
    let token = match state
        .token_cache
        .ensure_token_valid_for_streaming(db, &credential.uuid, 10)
        .await
    {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!(
                "[KIRO_STREAM] Token validation failed, loading from source: {}",
                e
            );
            // 回退到从源文件加载
            let mut kiro = KiroProvider::new();
            if let Err(e) = kiro.load_credentials_from_path(&creds_file_path).await {
                let _ = state.pool_service.mark_unhealthy(
                    db,
                    &credential.uuid,
                    Some(&format!("Failed to load credentials: {}", e)),
                );
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": {"message": format!("Failed to load Kiro credentials: {}", e)}})),
                )
                    .into_response();
            }
            if let Err(e) = kiro.refresh_token().await {
                let _ = state.pool_service.mark_unhealthy(
                    db,
                    &credential.uuid,
                    Some(&format!("Token refresh failed: {}", e)),
                );
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {}", e)}})),
                )
                    .into_response();
            }
            kiro.credentials.access_token.unwrap_or_default()
        }
    };

    // 创建 KiroProvider 并设置 token
    let mut kiro = KiroProvider::new();
    // 从源文件加载其他配置（region, profile_arn 等）
    // 注意：必须先加载凭证文件，再设置 token，因为 load_credentials_from_path 会覆盖整个 credentials
    let _ = kiro.load_credentials_from_path(&creds_file_path).await;
    // 使用缓存的 token 覆盖文件中的 token（缓存的 token 更新）
    kiro.credentials.access_token = Some(token);

    tracing::info!("[KIRO_STREAM] 准备调用 call_api_stream_anthropic (直接转换)");

    // 调用流式 API - 直接使用 Anthropic 格式（需求 4.1, 4.2, 4.3: 401/403 错误重试逻辑）
    let stream_response = match kiro.call_api_stream_anthropic(request).await {
        Ok(stream) => {
            tracing::info!("[KIRO_STREAM] call_api_stream 成功返回流");
            stream
        }
        Err(e) => {
            tracing::error!("[KIRO_STREAM] call_api_stream 失败: {}", e);
            // 检查是否是 401/403 错误或 Token 过期，需要刷新 token 重试（需求 4.1）
            let needs_token_refresh = matches!(
                &e,
                crate::providers::ProviderError::AuthenticationError(_)
                    | crate::providers::ProviderError::TokenExpired(_)
            );

            if needs_token_refresh {
                tracing::info!(
                    "[KIRO_STREAM] Got auth/token error ({}), forcing token refresh for {}",
                    e.short_message(),
                    &credential.uuid[..8]
                );
                // 强制刷新 token（需求 4.1）
                let new_token = match state
                    .token_cache
                    .refresh_and_cache(db, &credential.uuid, true)
                    .await
                {
                    Ok(t) => t,
                    Err(refresh_err) => {
                        // 需求 4.3: Token 刷新失败时返回明确的错误信息
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&format!("Token refresh failed: {}", refresh_err)),
                        );
                        return (
                            StatusCode::UNAUTHORIZED,
                            Json(serde_json::json!({
                                "error": {
                                    "type": "authentication_error",
                                    "message": format!("Token refresh failed: {}", refresh_err)
                                }
                            })),
                        )
                            .into_response();
                    }
                };

                // 使用新 token 重试（需求 4.2）
                kiro.credentials.access_token = Some(new_token);
                match kiro.call_api_stream_anthropic(request).await {
                    Ok(stream) => stream,
                    Err(retry_err) => {
                        let _ = state.pool_service.mark_unhealthy(
                            db,
                            &credential.uuid,
                            Some(&retry_err.to_string()),
                        );
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({
                                "error": {
                                    "type": "api_error",
                                    "message": format!("Retry failed after token refresh: {}", retry_err)
                                }
                            })),
                        )
                            .into_response();
                    }
                }
            } else {
                let _ =
                    state
                        .pool_service
                        .mark_unhealthy(db, &credential.uuid, Some(&e.to_string()));
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": {
                            "type": "api_error",
                            "message": e.to_string()
                        }
                    })),
                )
                    .into_response();
            }
        }
    };

    // 记录成功
    let _ = state
        .pool_service
        .mark_healthy(db, &credential.uuid, Some(&request.model));
    let _ = state.pool_service.record_usage(db, &credential.uuid);

    tracing::info!(
        "[KIRO_STREAM] 开始处理流式响应, model={}, flow_id={:?}",
        request.model,
        flow_id
    );

    // 使用新的统一流处理管道 (Kiro → Anthropic)
    let config = PipelineConfig::kiro_to_anthropic(request.model.clone());
    let pipeline = std::sync::Arc::new(tokio::sync::Mutex::new(StreamPipeline::new(config)));

    // 获取 flow_id 的克隆用于回调
    let flow_id_owned = flow_id.map(|s| s.to_string());
    let flow_monitor = state.flow_monitor.clone();

    // 创建转换流
    let pipeline_clone = pipeline.clone();
    let flow_id_for_stream = flow_id_owned.clone();
    let flow_monitor_for_stream = flow_monitor.clone();

    let pipeline_for_finalize = pipeline.clone();
    let flow_id_for_finalize = flow_id_owned.clone();
    let flow_monitor_for_finalize = flow_monitor.clone();

    let final_stream = async_stream::stream! {
        use futures::StreamExt;

        let mut stream_response = stream_response;

        while let Some(chunk_result) = stream_response.next().await {
            match chunk_result {
                Ok(bytes) => {
                    // 调试日志：记录接收到的字节数
                    tracing::debug!(
                        "[KIRO_STREAM] 收到 {} 字节数据",
                        bytes.len()
                    );

                    // 使用 Pipeline 处理字节块
                    let sse_strings = {
                        let mut pipeline_guard = pipeline_clone.lock().await;
                        pipeline_guard.process_chunk(&bytes)
                    };

                    // 调试日志：记录生成的 SSE 事件数量
                    tracing::debug!(
                        "[KIRO_STREAM] 生成 {} 个 SSE 事件",
                        sse_strings.len()
                    );

                    for sse_str in sse_strings {
                        // 调用 FlowMonitor.process_chunk()（需求 3.2）
                        if let Some(ref fid) = flow_id_for_stream {
                            // 解析 SSE 事件类型和数据
                            let lines: Vec<&str> = sse_str.lines().collect();
                            let mut event_type: Option<&str> = None;
                            let mut data: Option<&str> = None;

                            for line in &lines {
                                if line.starts_with("event: ") {
                                    event_type = Some(&line[7..]);
                                } else if line.starts_with("data: ") {
                                    data = Some(&line[6..]);
                                }
                            }

                            if let Some(d) = data {
                                let flow_monitor_clone = flow_monitor_for_stream.clone();
                                let fid_clone = fid.clone();
                                let event_type_owned = event_type.map(|s| s.to_string());
                                let data_owned = d.to_string();

                                tokio::spawn(async move {
                                    flow_monitor_clone
                                        .process_chunk(
                                            &fid_clone,
                                            event_type_owned.as_deref(),
                                            &data_owned,
                                        )
                                        .await;
                                });
                            }
                        }

                        // 立即 yield SSE 事件
                        yield Ok::<String, StreamError>(sse_str);
                    }
                }
                Err(e) => {
                    // 需求 5.1, 5.3: 流式传输期间发生错误时，发出错误事件并以失败状态完成 flow
                    tracing::error!("[KIRO_STREAM] 流式传输期间发生错误: {}", e);

                    // 根据 StreamError 类型映射到 FlowErrorType
                    let flow_error_type = match &e {
                        StreamError::Network(_) => FlowErrorType::Network,
                        StreamError::Timeout => FlowErrorType::Timeout,
                        StreamError::ProviderError { status, .. } => {
                            FlowErrorType::from_status_code(*status)
                        }
                        StreamError::ParseError(_) => FlowErrorType::Other,
                        StreamError::ClientDisconnected => FlowErrorType::Cancelled,
                        StreamError::BufferOverflow => FlowErrorType::Other,
                        StreamError::Internal(_) => FlowErrorType::ServerError,
                    };

                    // 调用 FlowMonitor.fail_flow() 标记失败
                    if let Some(ref fid) = flow_id_for_stream {
                        let flow_error = FlowError::new(
                            flow_error_type,
                            format!("流式传输错误: {}", e),
                        );
                        flow_monitor_for_stream.fail_flow(fid, flow_error).await;
                    }

                    // 发送 SSE 错误事件
                    yield Err(e);
                    return;
                }
            }
        }

        tracing::info!("[KIRO_STREAM] 流结束，生成 finalize 事件");

        // 流结束，使用 Pipeline 生成 finalize 事件
        let final_events = {
            let mut pipeline_guard = pipeline_for_finalize.lock().await;
            pipeline_guard.finish()
        };

        tracing::debug!("[KIRO_STREAM] finalize 生成 {} 个事件", final_events.len());

        for sse_str in final_events {
            // 调用 FlowMonitor.process_chunk()
            if let Some(ref fid) = flow_id_for_finalize {
                let lines: Vec<&str> = sse_str.lines().collect();
                let mut event_type: Option<&str> = None;
                let mut data: Option<&str> = None;

                for line in &lines {
                    if line.starts_with("event: ") {
                        event_type = Some(&line[7..]);
                    } else if line.starts_with("data: ") {
                        data = Some(&line[6..]);
                    }
                }

                if let Some(d) = data {
                    let flow_monitor_clone = flow_monitor_for_finalize.clone();
                    let fid_clone = fid.clone();
                    let event_type_owned = event_type.map(|s| s.to_string());
                    let data_owned = d.to_string();

                    tokio::spawn(async move {
                        flow_monitor_clone
                            .process_chunk(&fid_clone, event_type_owned.as_deref(), &data_owned)
                            .await;
                    });
                }
            }

            yield Ok::<String, StreamError>(sse_str);
        }
    };

    tracing::info!("[KIRO_STREAM] 构建 SSE 响应");

    // 转换为 Body 流
    let body_stream = final_stream.map(|result| -> Result<axum::body::Bytes, std::io::Error> {
        match result {
            Ok(event) => Ok(axum::body::Bytes::from(event)),
            Err(e) => Ok(axum::body::Bytes::from(e.to_sse_error())),
        }
    });

    // 构建 SSE 响应
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .header(header::TRANSFER_ENCODING, "chunked")
        .header("X-Accel-Buffering", "no")
        .body(Body::from_stream(body_stream))
        .unwrap_or_else(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(
                    serde_json::json!({"error": {"message": "Failed to build streaming response"}}),
                ),
            )
                .into_response()
        })
}

/// 解析 Antigravity 累积的流式响应数据
///
/// Antigravity 返回的流式数据是分片的 JSON，格式如下：
/// ```json
/// {
///   "response": {
///     "candidates": [{
///       "content": {
///         "role": "model",
///         "parts": [
///           { "text": "..." },
///           { "inlineData": { "mimeType": "image/jpeg", "data": "base64..." } }
///         ]
///       }
///     }]
///   }
/// }
/// ```
fn parse_antigravity_accumulated_response(data: &str, model: &str) -> Result<String, String> {
    eprintln!(
        "[ANTIGRAVITY_PARSE] 开始解析累积数据，大小: {} bytes",
        data.len()
    );

    // 保存原始数据到文件用于调试
    let debug_dir = dirs::home_dir()
        .map(|h| h.join(".proxycast/logs"))
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"));
    let _ = std::fs::create_dir_all(&debug_dir);
    let debug_file = debug_dir.join("antigravity_stream_raw.txt");
    let _ = std::fs::write(&debug_file, data);
    eprintln!("[ANTIGRAVITY_PARSE] 原始数据已保存到: {:?}", debug_file);

    // 打印数据的前1000字符用于调试
    eprintln!(
        "[ANTIGRAVITY_PARSE] 数据前1000字符:\n{}",
        &data[..data.len().min(1000)]
    );

    // 尝试解析 JSON
    // Antigravity 流式响应可能是多个 JSON 对象，每个对象一行
    // 或者是一个大的 JSON 对象

    // 首先尝试直接解析为单个 JSON
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
        eprintln!("[ANTIGRAVITY_PARSE] 单个 JSON 解析成功");
        return parse_antigravity_json(&json, model);
    }

    // 如果失败，尝试按行解析，找到包含 candidates 的 JSON
    eprintln!("[ANTIGRAVITY_PARSE] 单个 JSON 解析失败，尝试按行解析");

    let mut all_text = String::new();
    let mut all_images: Vec<(String, String)> = Vec::new(); // (mime_type, data)
    let mut found_any = false;

    for line in data.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // 尝试解析每一行
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some((text, images)) = extract_content_from_json(&json) {
                all_text.push_str(&text);
                all_images.extend(images);
                found_any = true;
            }
        }
    }

    if found_any {
        eprintln!(
            "[ANTIGRAVITY_PARSE] 按行解析成功，文本长度: {}, 图片数: {}",
            all_text.len(),
            all_images.len()
        );
        return build_sse_response(&all_text, &all_images, model);
    }

    // 如果还是失败，尝试找到 JSON 对象的边界
    eprintln!("[ANTIGRAVITY_PARSE] 按行解析失败，尝试查找 JSON 边界");

    // 查找所有 { 开头的位置，尝试解析
    let mut start = 0;
    while let Some(pos) = data[start..].find('{') {
        let json_start = start + pos;
        // 尝试从这个位置解析 JSON
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&data[json_start..]) {
            eprintln!("[ANTIGRAVITY_PARSE] 在位置 {} 找到有效 JSON", json_start);
            return parse_antigravity_json(&json, model);
        }
        start = json_start + 1;
        if start >= data.len() {
            break;
        }
    }

    Err(format!("无法解析响应数据，请查看 {:?}", debug_file))
}

/// 从 JSON 中提取内容
fn extract_content_from_json(json: &serde_json::Value) -> Option<(String, Vec<(String, String)>)> {
    // 尝试多种路径
    let candidates = json
        .get("response")
        .and_then(|r| r.get("candidates"))
        .or_else(|| json.get("candidates"))
        .and_then(|c| c.as_array())?;

    if candidates.is_empty() {
        return None;
    }

    let mut text = String::new();
    let mut images = Vec::new();

    for candidate in candidates {
        if let Some(parts) = candidate
            .get("content")
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.as_array())
        {
            for part in parts {
                if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                    text.push_str(t);
                }
                if let Some(inline_data) =
                    part.get("inlineData").or_else(|| part.get("inline_data"))
                {
                    if let Some(data) = inline_data.get("data").and_then(|d| d.as_str()) {
                        let mime = inline_data
                            .get("mimeType")
                            .or_else(|| inline_data.get("mime_type"))
                            .and_then(|m| m.as_str())
                            .unwrap_or("image/png");
                        images.push((mime.to_string(), data.to_string()));
                    }
                }
            }
        }
    }

    if text.is_empty() && images.is_empty() {
        None
    } else {
        Some((text, images))
    }
}

/// 解析 Antigravity JSON 响应
fn parse_antigravity_json(json: &serde_json::Value, model: &str) -> Result<String, String> {
    eprintln!(
        "[ANTIGRAVITY_PARSE] 解析 JSON，顶层类型: {}",
        if json.is_object() {
            "object"
        } else if json.is_array() {
            "array"
        } else {
            "other"
        }
    );

    if let Some(obj) = json.as_object() {
        eprintln!(
            "[ANTIGRAVITY_PARSE] 顶层 keys: {:?}",
            obj.keys().collect::<Vec<_>>()
        );
    }

    if let Some((text, images)) = extract_content_from_json(json) {
        return build_sse_response(&text, &images, model);
    }

    // 如果是数组，尝试处理每个元素
    if let Some(arr) = json.as_array() {
        eprintln!("[ANTIGRAVITY_PARSE] 顶层是数组，长度: {}", arr.len());
        let mut all_text = String::new();
        let mut all_images = Vec::new();

        for item in arr {
            if let Some((text, images)) = extract_content_from_json(item) {
                all_text.push_str(&text);
                all_images.extend(images);
            }
        }

        if !all_text.is_empty() || !all_images.is_empty() {
            return build_sse_response(&all_text, &all_images, model);
        }
    }

    Err("响应中没有 candidates".to_string())
}

/// 构建 SSE 响应
fn build_sse_response(
    text: &str,
    images: &[(String, String)],
    model: &str,
) -> Result<String, String> {
    let mut content = text.to_string();

    // 添加图片
    for (mime, data) in images {
        let image_url = format!("data:{};base64,{}", mime, data);
        content.push_str(&format!("\n\n![Generated Image]({})", image_url));
    }

    eprintln!("[ANTIGRAVITY_PARSE] 构建 SSE，内容长度: {}", content.len());

    let chunk_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();

    let mut sse_output = String::new();

    if !content.is_empty() {
        let content_chunk = serde_json::json!({
            "id": &chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": { "content": content },
                "finish_reason": serde_json::Value::Null
            }]
        });
        sse_output.push_str(&format!("data: {}\n\n", content_chunk.to_string()));
    }

    let done_chunk = serde_json::json!({
        "id": &chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    });
    sse_output.push_str(&format!("data: {}\n\n", done_chunk.to_string()));
    sse_output.push_str("data: [DONE]\n\n");

    Ok(sse_output)
}

/// 将 Gemini 流式响应 chunk 转换为 OpenAI SSE 格式
///
/// Gemini 流式响应格式:
/// ```json
/// {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"},"finishReason":"STOP"}]}
/// ```
///
/// OpenAI SSE 格式:
/// ```
/// data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}
/// ```
fn convert_gemini_chunk_to_openai_sse(json: &serde_json::Value, model: &str) -> Option<String> {
    // 检查是否有 candidates
    let candidates = json.get("candidates")?.as_array()?;
    if candidates.is_empty() {
        return None;
    }

    let candidate = &candidates[0];

    // 提取文本内容
    let mut content_delta: Option<String> = None;
    let mut has_image = false;
    let mut image_data: Option<String> = None;

    if let Some(content) = candidate.get("content") {
        if let Some(parts) = content.get("parts").and_then(|p| p.as_array()) {
            for part in parts {
                // 处理文本
                if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                    content_delta = Some(text.to_string());
                }

                // 处理图片（inlineData）
                if let Some(inline_data) =
                    part.get("inlineData").or_else(|| part.get("inline_data"))
                {
                    if let Some(data) = inline_data.get("data").and_then(|d| d.as_str()) {
                        let mime_type = inline_data
                            .get("mimeType")
                            .or_else(|| inline_data.get("mime_type"))
                            .and_then(|m| m.as_str())
                            .unwrap_or("image/png");

                        // 将图片作为 markdown 格式的 data URL
                        let image_url = format!("data:{};base64,{}", mime_type, data);
                        image_data = Some(format!("\n\n![Generated Image]({})", image_url));
                        has_image = true;
                    }
                }
            }
        }
    }

    // 检查 finish_reason
    let finish_reason = candidate
        .get("finishReason")
        .and_then(|f| f.as_str())
        .map(|r| match r {
            "STOP" => "stop",
            "MAX_TOKENS" => "length",
            "SAFETY" => "content_filter",
            "RECITATION" => "content_filter",
            _ => "stop",
        });

    // 如果没有内容变化且没有 finish_reason，跳过
    if content_delta.is_none() && !has_image && finish_reason.is_none() {
        return None;
    }

    // 合并文本和图片内容
    let final_content = match (content_delta, image_data) {
        (Some(text), Some(img)) => Some(format!("{}{}", text, img)),
        (Some(text), None) => Some(text),
        (None, Some(img)) => Some(img),
        (None, None) => None,
    };

    // 构建 OpenAI 格式的 delta
    let mut delta = serde_json::json!({});
    if let Some(content) = final_content {
        delta["content"] = serde_json::Value::String(content);
    }

    // 构建完整的 SSE 事件
    let chunk_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();

    let response = serde_json::json!({
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason
        }]
    });

    Some(format!("data: {}\n\n", response.to_string()))
}
