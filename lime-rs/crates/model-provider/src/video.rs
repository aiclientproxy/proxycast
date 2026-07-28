//! 视频 Provider 的唯一网络边界。

use crate::lowering::{
    build_fal_video_generation_body, build_xai_video_generation_body, ProtocolMappingError,
};
use reqwest::{Client, RequestBuilder, StatusCode};
use runtime_core::CanonicalRequest;
use serde_json::{json, Map, Value};
use std::fmt;
use std::time::Duration;

pub const DEFAULT_VIDEO_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
pub const DEFAULT_XAI_VIDEO_POLL_INTERVAL: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoProtocol {
    Fal,
    Xai,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoProviderConfig {
    pub protocol: VideoProtocol,
    pub endpoint: String,
    pub api_key: String,
    pub auth_header: String,
    pub auth_prefix: Option<String>,
    pub provider_id: Option<String>,
    pub poll_interval: Duration,
    pub overall_timeout: Duration,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VideoProviderRequest {
    pub model_id: String,
    pub request: CanonicalRequest,
    pub resume_request_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VideoProviderProgress {
    Started { request_id: String },
    Polling { request_id: String, status: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct VideoProviderOutput {
    pub video: Value,
    pub response: Value,
    pub provider_request_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoProviderError {
    pub code: &'static str,
    pub message: String,
    pub retryable: bool,
    pub stage: &'static str,
    pub status: Option<u16>,
    pub provider_code: Option<String>,
}

impl VideoProviderError {
    pub fn cancelled() -> Self {
        Self {
            code: "video_provider_cancelled",
            message: "视频任务已取消".to_string(),
            retryable: false,
            stage: "cancel",
            status: None,
            provider_code: None,
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.code == "video_provider_cancelled"
    }
}

impl fmt::Display for VideoProviderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for VideoProviderError {}

pub async fn execute_video_generation<OnProgress, IsCancelled>(
    config: &VideoProviderConfig,
    request: &VideoProviderRequest,
    mut on_progress: OnProgress,
    mut is_cancelled: IsCancelled,
) -> Result<VideoProviderOutput, VideoProviderError>
where
    OnProgress: FnMut(VideoProviderProgress) -> Result<(), String>,
    IsCancelled: FnMut() -> bool,
{
    validate_config(config)?;
    if is_cancelled() {
        return Err(VideoProviderError::cancelled());
    }
    let client = Client::builder()
        .no_proxy()
        .timeout(config.overall_timeout)
        .build()
        .map_err(|error| {
            provider_error(
                "video_provider_client_build_failed",
                format!("初始化视频 Provider client 失败: {error}"),
                false,
                "request",
                None,
                None,
            )
        })?;

    match config.protocol {
        VideoProtocol::Fal => execute_fal(&client, config, request).await,
        VideoProtocol::Xai => {
            execute_xai(
                &client,
                config,
                request,
                &mut on_progress,
                &mut is_cancelled,
            )
            .await
        }
    }
}

async fn execute_fal(
    client: &Client,
    config: &VideoProviderConfig,
    request: &VideoProviderRequest,
) -> Result<VideoProviderOutput, VideoProviderError> {
    if request.resume_request_id.is_some() {
        return Err(provider_error(
            "video_resume_protocol_mismatch",
            "Fal 同步视频协议不能恢复异步 request_id".to_string(),
            false,
            "resume",
            None,
            None,
        ));
    }
    let body = build_fal_video_generation_body(&request.model_id, &request.request)
        .map_err(mapping_error)?;
    let response = send_json(
        client.post(config.endpoint.trim()).json(&body),
        config,
        "video_provider_request_failed",
        "request",
    )
    .await?;
    let video = extract_generated_video(&response).ok_or_else(|| {
        provider_error(
            "video_result_empty",
            "视频服务未返回可用结果".to_string(),
            false,
            "result",
            None,
            None,
        )
    })?;
    Ok(VideoProviderOutput {
        video,
        response,
        provider_request_id: None,
    })
}

async fn execute_xai<OnProgress, IsCancelled>(
    client: &Client,
    config: &VideoProviderConfig,
    request: &VideoProviderRequest,
    on_progress: &mut OnProgress,
    is_cancelled: &mut IsCancelled,
) -> Result<VideoProviderOutput, VideoProviderError>
where
    OnProgress: FnMut(VideoProviderProgress) -> Result<(), String>,
    IsCancelled: FnMut() -> bool,
{
    let mut start_response = None;
    let request_id = match request
        .resume_request_id
        .as_deref()
        .map(str::trim)
        .filter(|request_id| !request_id.is_empty())
    {
        Some(request_id) => request_id.to_string(),
        None => {
            let body = build_xai_video_generation_body(&request.model_id, &request.request)
                .map_err(mapping_error)?;
            let response = send_json(
                client.post(config.endpoint.trim()).json(&body),
                config,
                "video_provider_start_failed",
                "start",
            )
            .await?;
            let request_id = response
                .get("request_id")
                .or_else(|| response.get("requestId"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|request_id| !request_id.is_empty())
                .ok_or_else(|| {
                    provider_error(
                        "video_provider_request_id_missing",
                        "xAI 视频服务未返回 request_id".to_string(),
                        false,
                        "start",
                        None,
                        None,
                    )
                })?
                .to_string();
            report_progress(
                on_progress,
                VideoProviderProgress::Started {
                    request_id: request_id.clone(),
                },
            )?;
            start_response = Some(response);
            request_id
        }
    };
    let poll_endpoint = xai_poll_endpoint(&config.endpoint, &request_id)?;
    let started = tokio::time::Instant::now();

    loop {
        if is_cancelled() {
            return Err(VideoProviderError::cancelled());
        }
        if started.elapsed() >= config.overall_timeout {
            return Err(provider_error(
                "video_provider_poll_timeout",
                format!("xAI 视频生成超时 (request_id={request_id})"),
                true,
                "poll",
                None,
                Some("timeout".to_string()),
            ));
        }
        tokio::time::sleep(config.poll_interval).await;
        if is_cancelled() {
            return Err(VideoProviderError::cancelled());
        }

        let poll_response = send_json(
            client.get(&poll_endpoint),
            config,
            "video_provider_poll_failed",
            "poll",
        )
        .await?;
        let status = poll_response
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        report_progress(
            on_progress,
            VideoProviderProgress::Polling {
                request_id: request_id.clone(),
                status: status.clone(),
            },
        )?;

        match status.as_str() {
            "done" => {
                let video = extract_generated_video(&poll_response).ok_or_else(|| {
                    provider_error(
                        "video_result_empty",
                        "xAI 视频任务完成但未返回视频 URL".to_string(),
                        false,
                        "result",
                        None,
                        Some("done".to_string()),
                    )
                })?;
                return Ok(VideoProviderOutput {
                    video,
                    response: json!({
                        "start": start_response,
                        "poll": poll_response,
                    }),
                    provider_request_id: Some(request_id),
                });
            }
            "failed" => {
                return Err(provider_error(
                    "video_provider_generation_failed",
                    format!("xAI 视频生成失败 (request_id={request_id})"),
                    false,
                    "poll",
                    None,
                    Some("failed".to_string()),
                ));
            }
            "expired" => {
                return Err(provider_error(
                    "video_provider_generation_expired",
                    format!("xAI 视频生成请求已过期 (request_id={request_id})"),
                    true,
                    "poll",
                    None,
                    Some("expired".to_string()),
                ));
            }
            _ => {}
        }
    }
}

fn validate_config(config: &VideoProviderConfig) -> Result<(), VideoProviderError> {
    if config.endpoint.trim().is_empty() {
        return Err(provider_error(
            "video_endpoint_missing",
            "视频服务 endpoint 不能为空".to_string(),
            false,
            "request",
            None,
            None,
        ));
    }
    if !config.api_key.trim().is_empty() && config.auth_header.trim().is_empty() {
        return Err(provider_error(
            "video_auth_header_missing",
            "视频服务鉴权 header 不能为空".to_string(),
            false,
            "request",
            None,
            None,
        ));
    }
    if config.poll_interval.is_zero() || config.overall_timeout.is_zero() {
        return Err(provider_error(
            "video_timeout_invalid",
            "视频服务 poll interval 和 timeout 必须大于零".to_string(),
            false,
            "request",
            None,
            None,
        ));
    }
    Ok(())
}

async fn send_json(
    request: RequestBuilder,
    config: &VideoProviderConfig,
    code: &'static str,
    stage: &'static str,
) -> Result<Value, VideoProviderError> {
    let response = apply_headers(request, config)
        .send()
        .await
        .map_err(|error| {
            provider_error(
                code,
                format!("请求视频服务失败: {error}"),
                true,
                stage,
                None,
                None,
            )
        })?;
    let status = response.status();
    let response_text = response.text().await.map_err(|error| {
        provider_error(
            "video_provider_response_read_error",
            format!("读取视频服务响应失败: {error}"),
            true,
            stage,
            Some(status.as_u16()),
            None,
        )
    })?;
    if !status.is_success() {
        return Err(provider_error(
            code,
            format!(
                "视频服务返回 HTTP {}: {}",
                status.as_u16(),
                summarize_provider_body(&response_text)
            ),
            status_is_retryable(status),
            stage,
            Some(status.as_u16()),
            Some(status.as_u16().to_string()),
        ));
    }
    serde_json::from_str(&response_text).map_err(|error| {
        provider_error(
            "video_provider_response_invalid",
            format!("解析视频服务响应失败: {error}"),
            false,
            stage,
            Some(status.as_u16()),
            None,
        )
    })
}

fn apply_headers(request: RequestBuilder, config: &VideoProviderConfig) -> RequestBuilder {
    let mut request = request;
    let api_key = config.api_key.trim();
    if !api_key.is_empty() {
        let auth_value = config
            .auth_prefix
            .as_deref()
            .map(str::trim)
            .filter(|prefix| !prefix.is_empty())
            .map(|prefix| format!("{prefix} {api_key}"))
            .unwrap_or_else(|| api_key.to_string());
        request = request.header(config.auth_header.trim(), auth_value);
    }
    if let Some(provider_id) = config
        .provider_id
        .as_deref()
        .map(str::trim)
        .filter(|provider_id| !provider_id.is_empty())
    {
        request = request.header("X-Provider-Id", provider_id);
    }
    request
}

fn xai_poll_endpoint(endpoint: &str, request_id: &str) -> Result<String, VideoProviderError> {
    if request_id.contains(['/', '\\']) || request_id == "." || request_id == ".." {
        return Err(provider_error(
            "video_provider_request_id_invalid",
            "xAI 视频 request_id 格式无效".to_string(),
            false,
            "poll",
            None,
            None,
        ));
    }
    let mut url = url::Url::parse(endpoint.trim()).map_err(|error| {
        provider_error(
            "video_endpoint_invalid",
            format!("xAI 视频 endpoint 无效: {error}"),
            false,
            "poll",
            None,
            None,
        )
    })?;
    let mut path = url.path().trim_end_matches('/').to_string();
    if let Some(prefix) = path.strip_suffix("/generations") {
        path = prefix.to_string();
    }
    path.push('/');
    path.push_str(request_id);
    url.set_path(&path);
    url.set_query(None);
    url.set_fragment(None);
    Ok(url.to_string())
}

fn extract_generated_video(response: &Value) -> Option<Value> {
    response
        .get("data")
        .and_then(Value::as_array)
        .and_then(|items| items.iter().find_map(extract_video_from_candidate))
        .or_else(|| response.get("video").and_then(extract_video_from_candidate))
        .or_else(|| extract_video_from_candidate(response))
}

fn extract_video_from_candidate(candidate: &Value) -> Option<Value> {
    let record = candidate.as_object()?;
    let url = record
        .get("url")
        .or_else(|| record.get("video_url"))
        .or_else(|| record.get("videoUrl"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|url| !url.is_empty())?;
    let mut video = Map::new();
    video.insert("url".to_string(), json!(url));
    for key in [
        "id",
        "mime_type",
        "mimeType",
        "duration",
        "width",
        "height",
        "thumbnail_url",
        "thumbnailUrl",
    ] {
        if let Some(value) = record.get(key) {
            video.insert(key.to_string(), value.clone());
        }
    }
    Some(Value::Object(video))
}

fn mapping_error(error: ProtocolMappingError) -> VideoProviderError {
    provider_error(
        "video_request_mapping_failed",
        format!("构建视频生成请求失败: {error}"),
        false,
        "request",
        None,
        None,
    )
}

fn report_progress<OnProgress>(
    on_progress: &mut OnProgress,
    progress: VideoProviderProgress,
) -> Result<(), VideoProviderError>
where
    OnProgress: FnMut(VideoProviderProgress) -> Result<(), String>,
{
    on_progress(progress).map_err(|error| {
        provider_error(
            "video_progress_persistence_failed",
            format!("持久化视频 Provider 进度失败: {error}"),
            true,
            "state",
            None,
            None,
        )
    })
}

fn status_is_retryable(status: StatusCode) -> bool {
    status.is_server_error()
        || status == StatusCode::REQUEST_TIMEOUT
        || status == StatusCode::TOO_MANY_REQUESTS
}

fn provider_error(
    code: &'static str,
    message: String,
    retryable: bool,
    stage: &'static str,
    status: Option<u16>,
    provider_code: Option<String>,
) -> VideoProviderError {
    VideoProviderError {
        code,
        message,
        retryable,
        stage,
        status,
        provider_code,
    }
}

fn summarize_provider_body(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.chars().count() <= 240 {
        return trimmed.to_string();
    }
    let summary: String = trimmed.chars().take(240).collect();
    format!("{summary}...")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xai_poll_url_replaces_generations_segment() {
        assert_eq!(
            xai_poll_endpoint("https://api.x.ai/v1/videos/generations", "request-123")
                .expect("poll endpoint"),
            "https://api.x.ai/v1/videos/request-123"
        );
    }

    #[test]
    fn extracts_xai_video_object() {
        assert_eq!(
            extract_generated_video(&json!({
                "status": "done",
                "video": { "url": "https://cdn.example.test/video.mp4" }
            }))
            .and_then(|video| video.get("url").cloned()),
            Some(json!("https://cdn.example.test/video.mp4"))
        );
    }
}
