//! 图像生成 API 处理器
//!
//! 实现 OpenAI 兼容的 `/v1/images/generations` 端点，
//! 通过 Antigravity Provider 调用 Gemini 图像生成模型。
//!
//! # 功能
//! - 接收 OpenAI 格式的图像生成请求
//! - 转换为 Antigravity/Gemini 格式
//! - 调用 Antigravity Provider
//! - 返回 OpenAI 格式的响应
//!
//! # 需求覆盖
//! - 需求 1.1: 实现 `/v1/images/generations` 端点
//! - 需求 4.1: 验证请求参数
//! - 需求 4.2: 获取 Antigravity 凭证
//! - 需求 4.3: 调用 Antigravity Provider
//! - 需求 4.4: 转换响应格式

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};

use crate::converter::openai_to_antigravity::{
    convert_antigravity_image_response, convert_image_request_to_antigravity,
};
use crate::models::openai::ImageGenerationRequest;
use crate::models::provider_pool_model::CredentialData;
use crate::providers::AntigravityProvider;
use crate::server::handlers::verify_api_key;
use crate::server::AppState;

/// 处理图像生成请求
///
/// # 端点
/// `POST /v1/images/generations`
///
/// # 请求格式
/// ```json
/// {
///   "prompt": "A cute cat",
///   "model": "dall-e-3",
///   "n": 1,
///   "size": "1024x1024",
///   "response_format": "url"
/// }
/// ```
///
/// # 响应格式
/// ```json
/// {
///   "created": 1234567890,
///   "data": [
///     {
///       "url": "data:image/png;base64,...",
///       "revised_prompt": "A cute fluffy cat"
///     }
///   ]
/// }
/// ```
pub async fn handle_image_generation(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<ImageGenerationRequest>,
) -> Response {
    // 验证 API Key
    if let Err(e) = verify_api_key(&headers, &state.api_key).await {
        return e.into_response();
    }

    // 验证请求参数
    if request.prompt.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": "prompt is required and cannot be empty",
                    "type": "invalid_request_error",
                    "code": "invalid_prompt"
                }
            })),
        )
            .into_response();
    }

    // 记录请求日志
    // 安全截取 prompt，避免 UTF-8 字符边界问题
    let prompt_preview: String = request.prompt.chars().take(50).collect();
    let prompt_display = if request.prompt.chars().count() > 50 {
        format!("{}...", prompt_preview)
    } else {
        request.prompt.clone()
    };
    state.logs.write().await.add(
        "info",
        &format!(
            "[IMAGE] 收到图像生成请求: model={}, prompt={}, n={}, response_format={}",
            request.model, prompt_display, request.n, request.response_format
        ),
    );

    // 获取 Antigravity 凭证
    let db = match &state.db {
        Some(db) => db,
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": "Database not available",
                        "type": "server_error"
                    }
                })),
            )
                .into_response();
        }
    };

    // 从凭证池获取 Antigravity 凭证
    let credential = match state
        .pool_service
        .select_credential(db, "antigravity", None)
    {
        Ok(Some(cred)) => cred,
        Ok(None) => {
            state
                .logs
                .write()
                .await
                .add("error", "[IMAGE] 没有可用的 Antigravity 凭证");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": {
                        "message": "No Antigravity credentials available for image generation",
                        "type": "server_error",
                        "code": "no_credentials"
                    }
                })),
            )
                .into_response();
        }
        Err(e) => {
            state
                .logs
                .write()
                .await
                .add("error", &format!("[IMAGE] 获取凭证失败: {}", e));
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Failed to get credentials: {}", e),
                        "type": "server_error"
                    }
                })),
            )
                .into_response();
        }
    };

    // 提取 Antigravity 凭证信息
    let (creds_file_path, project_id) = match &credential.credential {
        CredentialData::AntigravityOAuth {
            creds_file_path,
            project_id,
        } => (creds_file_path.clone(), project_id.clone()),
        _ => {
            state
                .logs
                .write()
                .await
                .add("error", "[IMAGE] 选中的凭证不是 Antigravity 类型");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": "Selected credential is not Antigravity type",
                        "type": "server_error"
                    }
                })),
            )
                .into_response();
        }
    };

    // 创建 Antigravity Provider
    let mut antigravity = AntigravityProvider::new();
    if let Err(e) = antigravity
        .load_credentials_from_path(&creds_file_path)
        .await
    {
        let _ = state.pool_service.mark_unhealthy(
            db,
            &credential.uuid,
            Some(&format!("Failed to load credentials: {}", e)),
        );
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": {
                    "message": format!("Failed to load Antigravity credentials: {}", e),
                    "type": "server_error"
                }
            })),
        )
            .into_response();
    }

    // 验证并刷新 Token
    let validation_result = antigravity.validate_token();
    if validation_result.needs_refresh() {
        tracing::info!("[IMAGE] Token 需要刷新，开始刷新...");
        if let Err(refresh_error) = antigravity.refresh_token_with_retry(3).await {
            tracing::error!("[IMAGE] Token 刷新失败: {:?}", refresh_error);
            let _ = state.pool_service.mark_unhealthy_with_details(
                db,
                &credential.uuid,
                &refresh_error,
            );
            let (status, message) = if refresh_error.requires_reauth() {
                (StatusCode::UNAUTHORIZED, refresh_error.user_message())
            } else {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    refresh_error.user_message(),
                )
            };
            return (
                status,
                Json(serde_json::json!({
                    "error": {
                        "message": message,
                        "type": "authentication_error"
                    }
                })),
            )
                .into_response();
        }
    }

    // 设置项目 ID
    if let Some(pid) = project_id {
        antigravity.project_id = Some(pid);
    } else if let Err(e) = antigravity.discover_project().await {
        tracing::warn!("[IMAGE] Failed to discover project: {}", e);
    }

    let proj_id = antigravity.project_id.clone().unwrap_or_default();

    // 转换请求为 Antigravity 格式
    let antigravity_request = convert_image_request_to_antigravity(&request, &proj_id);

    state.logs.write().await.add(
        "debug",
        &format!(
            "[IMAGE] Antigravity 请求: model={}",
            antigravity_request["model"].as_str().unwrap_or("unknown")
        ),
    );

    // 调用 Antigravity API - 直接使用 call_api 而不是 generate_content
    // 因为 generate_content 内部的 to_gemini_response 会丢失嵌套在 response 字段下的数据
    let model = antigravity_request["model"]
        .as_str()
        .unwrap_or("gemini-3-pro-image-preview");

    eprintln!("[IMAGE] 调用 Antigravity API: model={}", model);
    eprintln!(
        "[IMAGE] 请求内容: {}",
        serde_json::to_string_pretty(&antigravity_request).unwrap_or_default()
    );

    match antigravity
        .call_api("generateContent", &antigravity_request)
        .await
    {
        Ok(resp) => {
            // 调试：打印原始响应
            eprintln!(
                "[IMAGE] Antigravity 原始响应: {}",
                serde_json::to_string_pretty(&resp).unwrap_or_default()
            );
            state.logs.write().await.add(
                "debug",
                &format!(
                    "[IMAGE] Antigravity 原始响应: {}",
                    serde_json::to_string(&resp).unwrap_or_default()
                ),
            );

            // 转换响应为 OpenAI 格式
            match convert_antigravity_image_response(&resp, &request.response_format) {
                Ok(image_response) => {
                    // 记录成功
                    let _ = state
                        .pool_service
                        .mark_healthy(db, &credential.uuid, Some(model));
                    let _ = state.pool_service.record_usage(db, &credential.uuid);

                    state.logs.write().await.add(
                        "info",
                        &format!("[IMAGE] 图像生成成功: {} 张图片", image_response.data.len()),
                    );

                    (StatusCode::OK, Json(image_response)).into_response()
                }
                Err(e) => {
                    state
                        .logs
                        .write()
                        .await
                        .add("error", &format!("[IMAGE] 响应转换失败: {}", e));
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({
                            "error": {
                                "message": e,
                                "type": "server_error",
                                "code": "image_generation_failed"
                            }
                        })),
                    )
                        .into_response()
                }
            }
        }
        Err(e) => {
            let _ = state
                .pool_service
                .mark_unhealthy(db, &credential.uuid, Some(&e.to_string()));
            state
                .logs
                .write()
                .await
                .add("error", &format!("[IMAGE] Antigravity API 调用失败: {}", e));
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Image generation failed: {}", e),
                        "type": "server_error",
                        "code": "api_error"
                    }
                })),
            )
                .into_response()
        }
    }
}
