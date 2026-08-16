//! OpenAI-compatible speech and transcription transport.
//!
//! The media runtime owns task state and files. This module owns only typed
//! request lowering plus the provider HTTP boundary.

use crate::lowering::{build_openai_audio_speech_body, ProtocolMappingError};
use reqwest::header::{HeaderName, HeaderValue, CONTENT_TYPE};
use reqwest::{Client, StatusCode};
use runtime_core::CanonicalRequest;
use serde_json::{json, Value};
use std::fmt;
use std::time::Duration;

const MAX_ATTEMPTS: usize = 3;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioProviderConfig {
    pub endpoint: String,
    pub api_key: String,
    pub auth_header: String,
    pub auth_prefix: Option<String>,
    pub timeout: Duration,
}

impl AudioProviderConfig {
    pub fn with_endpoint(endpoint: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            api_key: api_key.into(),
            auth_header: "Authorization".to_string(),
            auth_prefix: Some("Bearer".to_string()),
            timeout: DEFAULT_TIMEOUT,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechProviderRequest {
    pub model_id: String,
    pub input: String,
    pub voice: Option<String>,
    pub instructions: Option<String>,
    pub response_format: Option<String>,
    pub speed: Option<String>,
}

impl SpeechProviderRequest {
    pub fn canonical_request(&self) -> CanonicalRequest {
        let mut request = CanonicalRequest::text(&self.model_id, &self.input);
        if let Some(value) = self
            .voice
            .as_deref()
            .filter(|value| !value.trim().is_empty())
        {
            request
                .provider_options
                .insert("voice".to_string(), json!(value.trim()));
        }
        if let Some(value) = self
            .instructions
            .as_deref()
            .filter(|value| !value.trim().is_empty())
        {
            request
                .provider_options
                .insert("instructions".to_string(), json!(value.trim()));
        }
        if let Some(value) = self
            .response_format
            .as_deref()
            .filter(|value| !value.trim().is_empty())
        {
            request
                .provider_options
                .insert("response_format".to_string(), json!(value.trim()));
        }
        if let Some(value) = self
            .speed
            .as_deref()
            .filter(|value| !value.trim().is_empty())
        {
            request
                .provider_options
                .insert("speed".to_string(), json!(value.trim()));
        }
        request
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeechProviderOutput {
    pub audio: Vec<u8>,
    pub mime_type: String,
    pub response: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptionProviderRequest {
    pub model_id: String,
    pub audio: Vec<u8>,
    pub filename: String,
    pub mime_type: String,
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub response_format: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionProviderOutput {
    pub text: String,
    pub language: Option<String>,
    pub response: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioProviderError {
    pub code: &'static str,
    pub message: String,
    pub retryable: bool,
    pub stage: &'static str,
    pub status: Option<u16>,
    pub provider_code: Option<String>,
}

impl AudioProviderError {
    pub fn cancelled() -> Self {
        Self {
            code: "audio_provider_cancelled",
            message: "音频任务已取消".to_string(),
            retryable: false,
            stage: "cancel",
            status: None,
            provider_code: None,
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.code == "audio_provider_cancelled"
    }
}

impl fmt::Display for AudioProviderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for AudioProviderError {}

pub async fn execute_speech_generation(
    config: &AudioProviderConfig,
    request: &SpeechProviderRequest,
) -> Result<SpeechProviderOutput, AudioProviderError> {
    validate_config(config, "语音合成")?;
    if request.model_id.trim().is_empty() || request.input.trim().is_empty() {
        return Err(audio_error(
            "speech_invalid_request",
            "语音合成需要非空 model 和 input".to_string(),
            false,
            "lowering",
            None,
            None,
        ));
    }
    let canonical = request.canonical_request();
    let body =
        build_openai_audio_speech_body(&request.model_id, &canonical).map_err(mapping_error)?;
    let client = build_client(config)?;

    for attempt in 1..=MAX_ATTEMPTS {
        let response = apply_auth(client.post(&config.endpoint).json(&body), config)
            .send()
            .await
            .map_err(|error| {
                audio_error(
                    "speech_provider_request_failed",
                    format!("请求语音合成服务失败: {error}"),
                    attempt < MAX_ATTEMPTS,
                    "request",
                    None,
                    None,
                )
            })?;
        let status = response.status();
        if status.is_success() {
            let mime_type = response
                .headers()
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.split(';').next())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToString::to_string)
                .unwrap_or_else(|| speech_mime_type(request.response_format.as_deref()));
            let audio = response.bytes().await.map_err(|error| {
                audio_error(
                    "speech_provider_response_read_failed",
                    format!("读取语音合成结果失败: {error}"),
                    true,
                    "result",
                    Some(status.as_u16()),
                    None,
                )
            })?;
            if audio.is_empty() {
                return Err(audio_error(
                    "speech_result_empty",
                    "语音合成服务返回空音频".to_string(),
                    false,
                    "result",
                    Some(status.as_u16()),
                    None,
                ));
            }
            return Ok(SpeechProviderOutput {
                response: json!({
                    "status": status.as_u16(),
                    "contentType": mime_type,
                    "byteLength": audio.len(),
                }),
                audio: audio.to_vec(),
                mime_type,
            });
        }
        let (message, provider_code) = error_body(response).await;
        if is_retryable_status(status) && attempt < MAX_ATTEMPTS {
            tokio::time::sleep(retry_delay(attempt)).await;
            continue;
        }
        return Err(audio_error(
            "speech_provider_request_failed",
            format!("语音合成服务返回 HTTP {}: {message}", status.as_u16()),
            is_retryable_status(status),
            "request",
            Some(status.as_u16()),
            provider_code,
        ));
    }
    unreachable!("speech attempts are bounded")
}

pub async fn execute_transcription(
    config: &AudioProviderConfig,
    request: &TranscriptionProviderRequest,
) -> Result<TranscriptionProviderOutput, AudioProviderError> {
    validate_config(config, "音频转写")?;
    if request.model_id.trim().is_empty() || request.audio.is_empty() {
        return Err(audio_error(
            "transcription_invalid_request",
            "音频转写需要非空 model 和 audio".to_string(),
            false,
            "lowering",
            None,
            None,
        ));
    }
    let client = build_client(config)?;
    for attempt in 1..=MAX_ATTEMPTS {
        let filename = if request.filename.trim().is_empty() {
            "audio.bin".to_string()
        } else {
            request.filename.trim().to_string()
        };
        let file = reqwest::multipart::Part::bytes(request.audio.clone())
            .file_name(filename)
            .mime_str(if request.mime_type.trim().is_empty() {
                "application/octet-stream"
            } else {
                request.mime_type.trim()
            })
            .map_err(|error| {
                audio_error(
                    "transcription_source_mime_invalid",
                    format!("转写音频 MIME 无效: {error}"),
                    false,
                    "lowering",
                    None,
                    None,
                )
            })?;
        let mut form = reqwest::multipart::Form::new()
            .text("model", request.model_id.trim().to_string())
            .part("file", file);
        if let Some(value) = request
            .language
            .as_deref()
            .filter(|value| !value.trim().is_empty())
        {
            form = form.text("language", value.trim().to_string());
        }
        if let Some(value) = request
            .prompt
            .as_deref()
            .filter(|value| !value.trim().is_empty())
        {
            form = form.text("prompt", value.trim().to_string());
        }
        if let Some(value) = request
            .response_format
            .as_deref()
            .filter(|value| !value.trim().is_empty())
        {
            form = form.text("response_format", value.trim().to_string());
        }
        let response = apply_auth(client.post(&config.endpoint).multipart(form), config)
            .send()
            .await
            .map_err(|error| {
                audio_error(
                    "transcription_provider_request_failed",
                    format!("请求音频转写服务失败: {error}"),
                    attempt < MAX_ATTEMPTS,
                    "request",
                    None,
                    None,
                )
            })?;
        let status = response.status();
        if status.is_success() {
            let body = response.bytes().await.map_err(|error| {
                audio_error(
                    "transcription_provider_response_read_failed",
                    format!("读取音频转写结果失败: {error}"),
                    true,
                    "result",
                    Some(status.as_u16()),
                    None,
                )
            })?;
            return parse_transcription_response(&body, request.response_format.as_deref());
        }
        let (message, provider_code) = error_body(response).await;
        if is_retryable_status(status) && attempt < MAX_ATTEMPTS {
            tokio::time::sleep(retry_delay(attempt)).await;
            continue;
        }
        return Err(audio_error(
            "transcription_provider_request_failed",
            format!("音频转写服务返回 HTTP {}: {message}", status.as_u16()),
            is_retryable_status(status),
            "request",
            Some(status.as_u16()),
            provider_code,
        ));
    }
    unreachable!("transcription attempts are bounded")
}

fn parse_transcription_response(
    body: &[u8],
    response_format: Option<&str>,
) -> Result<TranscriptionProviderOutput, AudioProviderError> {
    if body.is_empty() {
        return Err(audio_error(
            "transcription_provider_empty_response",
            "音频转写服务返回空结果".to_string(),
            false,
            "result",
            None,
            None,
        ));
    }
    let should_parse_json = response_format
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "json" | "verbose_json"
            )
        })
        .unwrap_or(false)
        || body.first() == Some(&b'{');
    if !should_parse_json {
        let text = String::from_utf8_lossy(body).trim().to_string();
        if text.is_empty() {
            return Err(audio_error(
                "transcription_provider_empty_response",
                "音频转写服务返回空文本".to_string(),
                false,
                "result",
                None,
                None,
            ));
        }
        return Ok(TranscriptionProviderOutput {
            text,
            language: None,
            response: json!({ "text": String::from_utf8_lossy(body).trim() }),
        });
    }
    let value: Value = serde_json::from_slice(body).map_err(|error| {
        audio_error(
            "transcription_response_invalid",
            format!("解析音频转写 JSON 失败: {error}"),
            false,
            "result",
            None,
            None,
        )
    })?;
    let text = value
        .get("text")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            audio_error(
            "transcription_provider_empty_response",
                "音频转写 JSON 缺少 text".to_string(),
                false,
                "result",
                None,
                None,
            )
        })?
        .to_string();
    Ok(TranscriptionProviderOutput {
        text,
        language: value
            .get("language")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        response: value,
    })
}

fn validate_config(config: &AudioProviderConfig, label: &str) -> Result<(), AudioProviderError> {
    if config.endpoint.trim().is_empty() {
        return Err(audio_error(
            "audio_endpoint_missing",
            format!("{label} endpoint 不能为空"),
            false,
            "request",
            None,
            None,
        ));
    }
    reqwest::Url::parse(config.endpoint.trim()).map_err(|error| {
        audio_error(
            "audio_endpoint_invalid",
            format!("{label} endpoint 无效: {error}"),
            false,
            "request",
            None,
            None,
        )
    })?;
    if config.timeout.is_zero() {
        return Err(audio_error(
            "audio_timeout_invalid",
            format!("{label} timeout 必须大于零"),
            false,
            "request",
            None,
            None,
        ));
    }
    Ok(())
}

fn build_client(config: &AudioProviderConfig) -> Result<Client, AudioProviderError> {
    Client::builder()
        .no_proxy()
        .timeout(config.timeout)
        .build()
        .map_err(|error| {
            audio_error(
                "audio_provider_client_build_failed",
                format!("初始化音频 Provider client 失败: {error}"),
                false,
                "request",
                None,
                None,
            )
        })
}

fn apply_auth(
    request: reqwest::RequestBuilder,
    config: &AudioProviderConfig,
) -> reqwest::RequestBuilder {
    let api_key = config.api_key.trim();
    if api_key.is_empty() {
        return request;
    }
    let value = match config
        .auth_prefix
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        Some(prefix) => format!("{prefix} {api_key}"),
        None => api_key.to_string(),
    };
    let Ok(name) = HeaderName::from_bytes(config.auth_header.trim().as_bytes()) else {
        return request;
    };
    let Ok(value) = HeaderValue::from_str(&value) else {
        return request;
    };
    request.header(name, value)
}

async fn error_body(response: reqwest::Response) -> (String, Option<String>) {
    let status = response.status();
    let text = response.text().await.unwrap_or_default();
    let value = serde_json::from_str::<Value>(&text).ok();
    let message = value
        .as_ref()
        .and_then(|value| value.get("error"))
        .and_then(|error| error.get("message").or_else(|| error.get("detail")))
        .and_then(Value::as_str)
        .or_else(|| {
            value
                .as_ref()
                .and_then(|value| value.get("message"))
                .and_then(Value::as_str)
        })
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .unwrap_or_else(|| text.trim().to_string());
    let code = value
        .as_ref()
        .and_then(|value| value.get("error"))
        .and_then(|error| error.get("code").or_else(|| error.get("type")))
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .or_else(|| Some(format!("http_{}", status.as_u16())));
    (
        if message.is_empty() {
            "未提供错误详情".to_string()
        } else {
            message
        },
        code,
    )
}

fn is_retryable_status(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

fn retry_delay(attempt: usize) -> Duration {
    Duration::from_millis(200_u64.saturating_mul(1_u64 << attempt.min(4)))
}

fn speech_mime_type(response_format: Option<&str>) -> String {
    match response_format
        .unwrap_or("mp3")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "wav" => "audio/wav",
        "opus" => "audio/opus",
        "aac" => "audio/aac",
        "flac" => "audio/flac",
        "pcm" => "audio/pcm",
        _ => "audio/mpeg",
    }
    .to_string()
}

fn mapping_error(error: ProtocolMappingError) -> AudioProviderError {
    audio_error(
        "speech_request_mapping_failed",
        error.to_string(),
        false,
        "lowering",
        None,
        None,
    )
}

fn audio_error(
    code: &'static str,
    message: String,
    retryable: bool,
    stage: &'static str,
    status: Option<u16>,
    provider_code: Option<String>,
) -> AudioProviderError {
    AudioProviderError {
        code,
        message,
        retryable,
        stage,
        status,
        provider_code,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_json_transcription() {
        let output = parse_transcription_response(
            br#"{"text":"hello","language":"en","segments":[]}"#,
            Some("json"),
        )
        .expect("transcription");
        assert_eq!(output.text, "hello");
        assert_eq!(output.language.as_deref(), Some("en"));
    }

    #[test]
    fn maps_speech_format_to_mime() {
        assert_eq!(speech_mime_type(Some("wav")), "audio/wav");
        assert_eq!(speech_mime_type(Some("mp3")), "audio/mpeg");
    }
}
