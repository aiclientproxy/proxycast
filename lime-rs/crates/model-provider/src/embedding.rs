//! OpenAI-compatible embeddings transport.
//!
//! Cloud embedding requests cross the provider boundary here. Local ONNX
//! execution remains owned by the `embedding` crate and must not use this
//! module.

use crate::lowering::build_openai_embeddings_body;
use reqwest::header::{HeaderName, HeaderValue};
use reqwest::{Client, StatusCode};
use serde_json::Value;
use std::fmt;
use std::time::Duration;

const MAX_ATTEMPTS: usize = 3;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingProviderConfig {
    pub endpoint: String,
    pub api_key: String,
    pub auth_header: String,
    pub auth_prefix: Option<String>,
    pub timeout: Duration,
}

impl EmbeddingProviderConfig {
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
pub struct EmbeddingProviderRequest {
    pub model_id: String,
    pub inputs: Vec<String>,
    pub dimensions: Option<u32>,
    pub encoding_format: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingProviderOutput {
    /// Embeddings are returned in request order, regardless of wire indices.
    pub embeddings: Vec<Vec<f32>>,
    pub response: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingProviderError {
    pub code: &'static str,
    pub message: String,
    pub retryable: bool,
    pub stage: &'static str,
    pub status: Option<u16>,
    pub provider_code: Option<String>,
}

impl EmbeddingProviderError {
    pub fn cancelled() -> Self {
        Self {
            code: "embedding_provider_cancelled",
            message: "embedding 任务已取消".to_string(),
            retryable: false,
            stage: "cancel",
            status: None,
            provider_code: None,
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.code == "embedding_provider_cancelled"
    }
}

impl fmt::Display for EmbeddingProviderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for EmbeddingProviderError {}

pub async fn execute_embeddings(
    config: &EmbeddingProviderConfig,
    request: &EmbeddingProviderRequest,
) -> Result<EmbeddingProviderOutput, EmbeddingProviderError> {
    validate_config(config)?;
    if request.model_id.trim().is_empty() {
        return Err(embedding_error(
            "embedding_invalid_request",
            "embedding 需要非空 model".to_string(),
            false,
            "lowering",
            None,
            None,
        ));
    }
    if request.inputs.is_empty() || request.inputs.iter().all(|input| input.trim().is_empty()) {
        return Err(embedding_error(
            "embedding_invalid_request",
            "embedding 需要至少一个非空 input".to_string(),
            false,
            "lowering",
            None,
            None,
        ));
    }
    if request
        .encoding_format
        .as_deref()
        .is_some_and(|format| !format.trim().is_empty() && !format.eq_ignore_ascii_case("float"))
    {
        return Err(embedding_error(
            "embedding_encoding_unsupported",
            "当前 embedding 执行链只支持 encoding_format=float".to_string(),
            false,
            "lowering",
            None,
            None,
        ));
    }

    let body = build_openai_embeddings_body(
        &request.model_id,
        &request.inputs,
        request.dimensions,
        request.encoding_format.as_deref(),
    )
    .map_err(|message| {
        embedding_error(
            "embedding_request_mapping_failed",
            message,
            false,
            "lowering",
            None,
            None,
        )
    })?;
    let client = build_client(config)?;

    for attempt in 1..=MAX_ATTEMPTS {
        let response = apply_auth(client.post(&config.endpoint).json(&body), config)
            .send()
            .await
            .map_err(|error| {
                embedding_error(
                    "embedding_provider_request_failed",
                    format!("请求 embedding 服务失败: {error}"),
                    attempt < MAX_ATTEMPTS,
                    "request",
                    None,
                    None,
                )
            })?;
        let status = response.status();
        if status.is_success() {
            let body = response.bytes().await.map_err(|error| {
                embedding_error(
                    "embedding_provider_response_read_failed",
                    format!("读取 embedding 结果失败: {error}"),
                    true,
                    "result",
                    Some(status.as_u16()),
                    None,
                )
            })?;
            return parse_embeddings_response(&body, request.inputs.len());
        }

        let (message, provider_code) = error_body(response).await;
        if is_retryable_status(status) && attempt < MAX_ATTEMPTS {
            tokio::time::sleep(retry_delay(attempt)).await;
            continue;
        }
        return Err(embedding_error(
            "embedding_provider_request_failed",
            format!("embedding 服务返回 HTTP {}: {message}", status.as_u16()),
            is_retryable_status(status),
            "request",
            Some(status.as_u16()),
            provider_code,
        ));
    }

    unreachable!("embedding attempts are bounded")
}

fn parse_embeddings_response(
    body: &[u8],
    expected_count: usize,
) -> Result<EmbeddingProviderOutput, EmbeddingProviderError> {
    if body.is_empty() {
        return Err(embedding_error(
            "embedding_result_empty",
            "embedding 服务返回空结果".to_string(),
            false,
            "result",
            None,
            None,
        ));
    }
    let response: Value = serde_json::from_slice(body).map_err(|error| {
        embedding_error(
            "embedding_response_invalid",
            format!("解析 embedding JSON 失败: {error}"),
            false,
            "result",
            None,
            None,
        )
    })?;
    let data = response
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            embedding_error(
                "embedding_response_invalid",
                "embedding 响应缺少 data 数组".to_string(),
                false,
                "result",
                None,
                None,
            )
        })?;
    if data.len() != expected_count {
        return Err(embedding_error(
            "embedding_result_count_mismatch",
            format!(
                "embedding 响应数量 {} 与请求数量 {} 不一致",
                data.len(),
                expected_count
            ),
            false,
            "result",
            None,
            None,
        ));
    }

    let mut ordered = vec![None; expected_count];
    for (position, item) in data.iter().enumerate() {
        let embedding = item
            .get("embedding")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                embedding_error(
                    "embedding_vector_invalid",
                    format!("embedding data[{position}] 缺少 embedding 数组"),
                    false,
                    "result",
                    None,
                    None,
                )
            })?
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .and_then(|value| value.is_finite().then_some(value as f32))
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                embedding_error(
                    "embedding_vector_invalid",
                    format!("embedding data[{position}] 包含非有限数字"),
                    false,
                    "result",
                    None,
                    None,
                )
            })?;
        let index = item
            .get("index")
            .and_then(Value::as_u64)
            .map(|value| value as usize);
        let target = index.unwrap_or(position);
        if target >= expected_count || ordered[target].is_some() {
            return Err(embedding_error(
                "embedding_index_invalid",
                format!("embedding data[{position}] index 无效: {target}"),
                false,
                "result",
                None,
                None,
            ));
        }
        ordered[target] = Some(embedding);
    }
    let embeddings = ordered
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| {
            embedding.ok_or_else(|| {
                embedding_error(
                    "embedding_index_missing",
                    format!("embedding 响应缺少 index {index}"),
                    false,
                    "result",
                    None,
                    None,
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(EmbeddingProviderOutput {
        embeddings,
        response,
    })
}

fn validate_config(config: &EmbeddingProviderConfig) -> Result<(), EmbeddingProviderError> {
    if config.endpoint.trim().is_empty() {
        return Err(embedding_error(
            "embedding_endpoint_missing",
            "embedding endpoint 不能为空".to_string(),
            false,
            "request",
            None,
            None,
        ));
    }
    reqwest::Url::parse(config.endpoint.trim()).map_err(|error| {
        embedding_error(
            "embedding_endpoint_invalid",
            format!("embedding endpoint 无效: {error}"),
            false,
            "request",
            None,
            None,
        )
    })?;
    if config.timeout.is_zero() {
        return Err(embedding_error(
            "embedding_timeout_invalid",
            "embedding timeout 必须大于零".to_string(),
            false,
            "request",
            None,
            None,
        ));
    }
    Ok(())
}

fn build_client(config: &EmbeddingProviderConfig) -> Result<Client, EmbeddingProviderError> {
    Client::builder()
        .no_proxy()
        .timeout(config.timeout)
        .build()
        .map_err(|error| {
            embedding_error(
                "embedding_provider_client_build_failed",
                format!("初始化 embedding Provider client 失败: {error}"),
                false,
                "request",
                None,
                None,
            )
        })
}

fn apply_auth(
    request: reqwest::RequestBuilder,
    config: &EmbeddingProviderConfig,
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

fn embedding_error(
    code: &'static str,
    message: String,
    retryable: bool,
    stage: &'static str,
    status: Option<u16>,
    provider_code: Option<String>,
) -> EmbeddingProviderError {
    EmbeddingProviderError {
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
    fn parses_and_orders_batch_embeddings_by_index() {
        let output = parse_embeddings_response(
            br#"{"data":[{"index":1,"embedding":[2.0,3.0]},{"index":0,"embedding":[0.5,1.5]}],"usage":{"prompt_tokens":2}}"#,
            2,
        )
        .expect("embedding response");
        assert_eq!(output.embeddings, vec![vec![0.5, 1.5], vec![2.0, 3.0]]);
        assert_eq!(output.response["usage"]["prompt_tokens"], 2);
    }

    #[test]
    fn rejects_count_mismatch_and_duplicate_indices() {
        let mismatch = parse_embeddings_response(br#"{"data":[]}"#, 1).expect_err("count mismatch");
        assert_eq!(mismatch.code, "embedding_result_count_mismatch");

        let duplicate = parse_embeddings_response(
            br#"{"data":[{"index":0,"embedding":[1]},{"index":0,"embedding":[2]}]}"#,
            2,
        )
        .expect_err("duplicate index");
        assert_eq!(duplicate.code, "embedding_index_invalid");
    }

    #[test]
    fn rejects_non_float_encoding() {
        let request = EmbeddingProviderRequest {
            model_id: "model".to_string(),
            inputs: vec!["text".to_string()],
            dimensions: None,
            encoding_format: Some("base64".to_string()),
        };
        let error = tokio::runtime::Runtime::new()
            .expect("runtime")
            .block_on(execute_embeddings(
                &EmbeddingProviderConfig::with_endpoint("http://127.0.0.1:1", ""),
                &request,
            ))
            .expect_err("unsupported format");
        assert_eq!(error.code, "embedding_encoding_unsupported");
    }
}
