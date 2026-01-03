//! Antigravity Provider - Google 内部 Gemini 3 Pro 接口
//!
//! 支持 Gemini 3 Pro 等高级模型，通过 Google 内部 API 访问。

#![allow(dead_code)]

use super::traits::{CredentialProvider, ProviderResult};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::oneshot;
use uuid::Uuid;

// Constants
const ANTIGRAVITY_BASE_URL_DAILY: &str = "https://daily-cloudcode-pa.sandbox.googleapis.com";
const ANTIGRAVITY_BASE_URL_AUTOPUSH: &str = "https://autopush-cloudcode-pa.sandbox.googleapis.com";
const ANTIGRAVITY_API_VERSION: &str = "v1internal";
const CREDENTIALS_DIR: &str = ".antigravity";
const CREDENTIALS_FILE: &str = "oauth_creds.json";

// OAuth credentials - 与 Antigravity CLI 相同
pub const OAUTH_CLIENT_ID: &str =
    "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com";
pub const OAUTH_CLIENT_SECRET: &str = "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";

// OAuth scopes
const OAUTH_SCOPES: &[&str] = &[
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
];

// Token 刷新提前量（秒）
const REFRESH_SKEW: i64 = 3000;

// Token 即将过期的阈值（秒）- 10 分钟
const TOKEN_EXPIRING_SOON_THRESHOLD: i64 = 600;

/// Token 验证结果
/// Requirements: 1.1, 1.2, 1.3, 1.4
#[derive(Debug, Clone, PartialEq)]
pub enum TokenValidationResult {
    /// Token 有效，包含剩余有效时间（秒）
    Valid { expires_in_secs: i64 },
    /// Token 即将过期（少于 10 分钟），需要主动刷新
    ExpiringSoon { expires_in_secs: i64 },
    /// Token 已过期
    Expired,
    /// Token 无效（缺失、为空或格式错误）
    Invalid { reason: String },
}

impl TokenValidationResult {
    /// 是否需要刷新 Token
    pub fn needs_refresh(&self) -> bool {
        matches!(
            self,
            TokenValidationResult::ExpiringSoon { .. }
                | TokenValidationResult::Expired
                | TokenValidationResult::Invalid { .. }
        )
    }

    /// 是否可以使用（有效或即将过期但仍可用）
    pub fn is_usable(&self) -> bool {
        matches!(
            self,
            TokenValidationResult::Valid { .. } | TokenValidationResult::ExpiringSoon { .. }
        )
    }
}

/// Token 刷新错误类型
/// Requirements: 2.1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TokenRefreshError {
    /// OAuth invalid_grant 错误 - 需要用户重新授权
    InvalidGrant { message: String },
    /// 网络错误 - 可以重试
    NetworkError { message: String },
    /// 服务器错误 (5xx) - 可以重试
    ServerError { message: String },
    /// 未知错误
    Unknown { message: String },
}

impl TokenRefreshError {
    /// 是否需要用户重新授权
    pub fn requires_reauth(&self) -> bool {
        matches!(self, TokenRefreshError::InvalidGrant { .. })
    }

    /// 是否可以重试
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            TokenRefreshError::NetworkError { .. } | TokenRefreshError::ServerError { .. }
        )
    }

    /// 获取用户友好的错误消息
    pub fn user_message(&self) -> String {
        match self {
            TokenRefreshError::InvalidGrant { .. } => {
                "Antigravity 授权已过期，请重新登录授权".to_string()
            }
            TokenRefreshError::NetworkError { message } => {
                format!("网络连接失败: {}", message)
            }
            TokenRefreshError::ServerError { message } => {
                format!("Google 服务暂时不可用: {}", message)
            }
            TokenRefreshError::Unknown { message } => {
                format!("Token 刷新失败: {}", message)
            }
        }
    }
}

impl std::fmt::Display for TokenRefreshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.user_message())
    }
}

impl std::error::Error for TokenRefreshError {}

/// Antigravity 支持的模型列表
pub const ANTIGRAVITY_MODELS: &[&str] = &[
    "gemini-3-pro-preview",
    "gemini-3-pro-image-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-computer-use-preview-10-2025",
    "gemini-claude-sonnet-4-5",
    "gemini-claude-sonnet-4-5-thinking",
    "gemini-claude-opus-4-5-thinking",
];

/// 模型别名映射（用户友好名称 -> 内部名称）
fn alias_to_model_name(model: &str) -> &str {
    match model {
        "gemini-2.5-computer-use-preview-10-2025" => "rev19-uic3-1p",
        "gemini-3-pro-image-preview" => "gemini-3-pro-image",
        "gemini-3-pro-preview" => "gemini-3-pro-high",
        "gemini-3-flash-preview" => "gemini-3-flash",
        "gemini-2.5-flash" => "gemini-2.5-flash",
        "gemini-claude-sonnet-4-5" => "claude-sonnet-4-5",
        "gemini-claude-sonnet-4-5-thinking" => "claude-sonnet-4-5-thinking",
        "gemini-claude-opus-4-5-thinking" => "claude-opus-4-5-thinking",
        _ => model,
    }
}

/// 内部模型名称 -> 用户友好名称
#[allow(dead_code)]
fn model_name_to_alias(model: &str) -> &str {
    match model {
        "rev19-uic3-1p" => "gemini-2.5-computer-use-preview-10-2025",
        "gemini-3-pro-image" => "gemini-3-pro-image-preview",
        "gemini-3-pro-high" => "gemini-3-pro-preview",
        "gemini-3-flash" => "gemini-3-flash-preview",
        "gemini-2.5-flash" => "gemini-2.5-flash",
        "claude-sonnet-4-5" => "gemini-claude-sonnet-4-5",
        "claude-sonnet-4-5-thinking" => "gemini-claude-sonnet-4-5-thinking",
        "claude-opus-4-5-thinking" => "gemini-claude-opus-4-5-thinking",
        _ => model,
    }
}

/// 生成随机请求 ID
fn generate_request_id() -> String {
    format!("agent-{}", Uuid::new_v4())
}

/// 生成随机会话 ID
fn generate_session_id() -> String {
    // 使用 UUID 的一部分作为随机数
    let uuid = Uuid::new_v4();
    let bytes = uuid.as_bytes();
    let n: u64 = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]) % 9_000_000_000_000_000_000;
    format!("-{}", n)
}

/// 生成随机项目 ID
fn generate_project_id() -> String {
    let adjectives = ["useful", "bright", "swift", "calm", "bold"];
    let nouns = ["fuze", "wave", "spark", "flow", "core"];
    let uuid = Uuid::new_v4();
    let bytes = uuid.as_bytes();
    let adj = adjectives[(bytes[0] as usize) % adjectives.len()];
    let noun = nouns[(bytes[1] as usize) % nouns.len()];
    let random_part: String = uuid.to_string()[..5].to_lowercase();
    format!("{}-{}-{}", adj, noun, random_part)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntigravityCredentials {
    pub access_token: Option<String>,
    pub refresh_token: Option<String>,
    pub token_type: Option<String>,
    /// 过期时间戳（毫秒）- 兼容旧格式
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expiry_date: Option<i64>,
    /// 过期时间（RFC3339 格式）-
    pub expire: Option<String>,
    pub scope: Option<String>,
    /// 最后刷新时间（RFC3339 格式）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_refresh: Option<String>,
    /// 凭证类型标识
    #[serde(default = "default_antigravity_type", rename = "type")]
    pub cred_type: String,
    /// Token 有效期（秒）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_in: Option<i64>,
    /// Token 获取时间戳（毫秒）-
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
    /// 是否启用 -
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable: Option<bool>,
    /// 项目 ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    /// 用户邮箱
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
}

fn default_antigravity_type() -> String {
    "antigravity".to_string()
}

impl Default for AntigravityCredentials {
    fn default() -> Self {
        Self {
            access_token: None,
            refresh_token: None,
            token_type: Some("Bearer".to_string()),
            expiry_date: None,
            expire: None,
            scope: None,
            last_refresh: None,
            cred_type: default_antigravity_type(),
            expires_in: None,
            timestamp: None,
            enable: None,
            project_id: None,
            email: None,
        }
    }
}

/// Antigravity Provider
pub struct AntigravityProvider {
    pub credentials: AntigravityCredentials,
    pub project_id: Option<String>,
    pub client: Client,
    pub base_urls: Vec<String>,
    pub available_models: Vec<String>,
}

impl Default for AntigravityProvider {
    fn default() -> Self {
        Self {
            credentials: AntigravityCredentials::default(),
            project_id: None,
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .unwrap_or_else(|_| Client::new()),
            base_urls: vec![
                ANTIGRAVITY_BASE_URL_DAILY.to_string(),
                ANTIGRAVITY_BASE_URL_AUTOPUSH.to_string(),
            ],
            available_models: ANTIGRAVITY_MODELS.iter().map(|s| s.to_string()).collect(),
        }
    }
}

impl AntigravityProvider {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn default_creds_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(CREDENTIALS_DIR)
            .join(CREDENTIALS_FILE)
    }

    pub async fn load_credentials(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let path = Self::default_creds_path();

        if tokio::fs::try_exists(&path).await.unwrap_or(false) {
            let content = tokio::fs::read_to_string(&path).await?;
            let creds: AntigravityCredentials = serde_json::from_str(&content)?;
            self.credentials = creds;
        }

        Ok(())
    }

    pub async fn load_credentials_from_path(
        &mut self,
        path: &str,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let content = tokio::fs::read_to_string(path).await?;

        // 尝试解析为单个凭证对象
        if let Ok(creds) = serde_json::from_str::<AntigravityCredentials>(&content) {
            self.credentials = creds;
            // 如果凭证中有 project_id，设置到 provider
            if let Some(ref pid) = self.credentials.project_id {
                self.project_id = Some(pid.clone());
            }
            return Ok(());
        }

        // 尝试解析为数组格式（兼容 antigravity2api-nodejs 的 accounts.json）
        if let Ok(creds_array) = serde_json::from_str::<Vec<AntigravityCredentials>>(&content) {
            // 找到第一个启用的凭证
            if let Some(creds) = creds_array.into_iter().find(|c| c.enable != Some(false)) {
                self.credentials = creds;
                // 如果凭证中有 project_id，设置到 provider
                if let Some(ref pid) = self.credentials.project_id {
                    self.project_id = Some(pid.clone());
                }
                return Ok(());
            }
            return Err("凭证文件中没有可用的账号（所有账号都被禁用）".into());
        }

        Err("无法解析凭证文件，请确保是有效的 JSON 格式".into())
    }

    pub async fn save_credentials(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let path = Self::default_creds_path();
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let content = serde_json::to_string_pretty(&self.credentials)?;
        tokio::fs::write(&path, content).await?;
        Ok(())
    }

    pub fn is_token_valid(&self) -> bool {
        if self.credentials.access_token.is_none() {
            return false;
        }

        // 检查是否被禁用
        if self.credentials.enable == Some(false) {
            return false;
        }

        // 优先检查 RFC3339 格式的过期时间
        if let Some(expire_str) = &self.credentials.expire {
            if let Ok(expires) = chrono::DateTime::parse_from_rfc3339(expire_str) {
                let now = chrono::Utc::now();
                // Token valid if more than 5 minutes until expiry
                return expires > now + chrono::Duration::minutes(5);
            }
        }

        // 兼容旧的毫秒时间戳格式
        if let Some(expiry) = self.credentials.expiry_date {
            let now = chrono::Utc::now().timestamp_millis();
            // Token valid if more than 5 minutes until expiry
            return expiry > now + 300_000;
        }

        // 兼容 antigravity2api-nodejs 格式：timestamp + expires_in
        if let (Some(timestamp), Some(expires_in)) =
            (self.credentials.timestamp, self.credentials.expires_in)
        {
            let expiry = timestamp + (expires_in * 1000);
            let now = chrono::Utc::now().timestamp_millis();
            // Token valid if more than 5 minutes until expiry
            return expiry > now + 300_000;
        }

        true
    }

    pub fn is_token_expiring_soon(&self) -> bool {
        // 优先检查 RFC3339 格式的过期时间
        if let Some(expire_str) = &self.credentials.expire {
            if let Ok(expires) = chrono::DateTime::parse_from_rfc3339(expire_str) {
                let now = chrono::Utc::now();
                let refresh_skew = chrono::Duration::seconds(REFRESH_SKEW);
                return expires <= now + refresh_skew;
            }
        }

        // 兼容旧的毫秒时间戳格式
        if let Some(expiry) = self.credentials.expiry_date {
            let now = chrono::Utc::now().timestamp_millis();
            let refresh_skew_ms = REFRESH_SKEW * 1000;
            return expiry <= now + refresh_skew_ms;
        }

        // 兼容 antigravity2api-nodejs 格式：timestamp + expires_in
        if let (Some(timestamp), Some(expires_in)) =
            (self.credentials.timestamp, self.credentials.expires_in)
        {
            let expiry = timestamp + (expires_in * 1000);
            let now = chrono::Utc::now().timestamp_millis();
            let refresh_skew_ms = REFRESH_SKEW * 1000;
            return expiry <= now + refresh_skew_ms;
        }

        true
    }

    /// 验证 Token 状态（支持多种时间格式）
    /// Requirements: 1.1, 1.2, 1.3, 1.4
    pub fn validate_token(&self) -> TokenValidationResult {
        // 检查 access_token 是否存在且非空
        match &self.credentials.access_token {
            None => {
                return TokenValidationResult::Invalid {
                    reason: "access_token 缺失".to_string(),
                };
            }
            Some(token) if token.trim().is_empty() => {
                return TokenValidationResult::Invalid {
                    reason: "access_token 为空".to_string(),
                };
            }
            _ => {}
        }

        // 检查是否被禁用
        if self.credentials.enable == Some(false) {
            return TokenValidationResult::Invalid {
                reason: "凭证已被禁用".to_string(),
            };
        }

        // 检查 refresh_token 是否存在（用于后续刷新）
        if self.credentials.refresh_token.is_none() {
            return TokenValidationResult::Invalid {
                reason: "refresh_token 缺失，无法刷新".to_string(),
            };
        }

        let now = chrono::Utc::now();
        let now_millis = now.timestamp_millis();

        // 尝试解析过期时间（支持多种格式）
        let expires_in_secs: Option<i64> = {
            // 优先检查 RFC3339 格式
            if let Some(expire_str) = &self.credentials.expire {
                if let Ok(expires) = chrono::DateTime::parse_from_rfc3339(expire_str) {
                    Some((expires.timestamp_millis() - now_millis) / 1000)
                } else {
                    // RFC3339 解析失败，尝试其他格式
                    None
                }
            } else {
                None
            }
        }
        .or_else(|| {
            // 兼容毫秒时间戳格式
            self.credentials
                .expiry_date
                .map(|expiry| (expiry - now_millis) / 1000)
        })
        .or_else(|| {
            // 兼容 timestamp + expires_in 格式
            match (self.credentials.timestamp, self.credentials.expires_in) {
                (Some(timestamp), Some(expires_in)) => {
                    let expiry = timestamp + (expires_in * 1000);
                    Some((expiry - now_millis) / 1000)
                }
                _ => None,
            }
        });

        match expires_in_secs {
            Some(secs) if secs <= 0 => TokenValidationResult::Expired,
            Some(secs) if secs <= TOKEN_EXPIRING_SOON_THRESHOLD => {
                TokenValidationResult::ExpiringSoon {
                    expires_in_secs: secs,
                }
            }
            Some(secs) => TokenValidationResult::Valid {
                expires_in_secs: secs,
            },
            None => {
                // 无法解析过期时间，视为已过期（Requirements: 1.4）
                TokenValidationResult::Invalid {
                    reason: "无法解析过期时间格式".to_string(),
                }
            }
        }
    }

    /// 分类 Token 刷新错误
    /// Requirements: 2.1
    fn classify_refresh_error(status: u16, body: &str) -> TokenRefreshError {
        // 检查是否是 invalid_grant 错误
        if status == 400 && body.contains("invalid_grant") {
            return TokenRefreshError::InvalidGrant {
                message: "Refresh token 已失效或被撤销".to_string(),
            };
        }

        // 服务器错误 (5xx)
        if status >= 500 {
            return TokenRefreshError::ServerError {
                message: format!("HTTP {}: {}", status, body),
            };
        }

        // 其他客户端错误
        if status >= 400 {
            return TokenRefreshError::Unknown {
                message: format!("HTTP {}: {}", status, body),
            };
        }

        TokenRefreshError::Unknown {
            message: body.to_string(),
        }
    }

    /// 带重试的 Token 刷新
    /// Requirements: 2.2, 2.3
    pub async fn refresh_token_with_retry(
        &mut self,
        max_retries: u32,
    ) -> Result<String, TokenRefreshError> {
        let refresh_token = self
            .credentials
            .refresh_token
            .as_ref()
            .ok_or_else(|| TokenRefreshError::InvalidGrant {
                message: "No refresh token available".to_string(),
            })?
            .clone();

        let params = [
            ("client_id", OAUTH_CLIENT_ID),
            ("client_secret", OAUTH_CLIENT_SECRET),
            ("refresh_token", refresh_token.as_str()),
            ("grant_type", "refresh_token"),
        ];

        let mut last_error: Option<TokenRefreshError> = None;
        let mut retry_count = 0;

        while retry_count <= max_retries {
            if retry_count > 0 {
                // 指数退避：100ms, 200ms, 400ms, ...
                let delay_ms = 100 * (1 << (retry_count - 1));
                tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
                tracing::info!(
                    "[Antigravity] Token 刷新重试 {}/{}, 延迟 {}ms",
                    retry_count,
                    max_retries,
                    delay_ms
                );
            }

            let result = self
                .client
                .post("https://oauth2.googleapis.com/token")
                .form(&params)
                .send()
                .await;

            match result {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        // 成功，解析响应
                        match resp.json::<serde_json::Value>().await {
                            Ok(data) => {
                                let new_token = data["access_token"].as_str().ok_or_else(|| {
                                    TokenRefreshError::Unknown {
                                        message: "响应中缺少 access_token".to_string(),
                                    }
                                })?;

                                self.credentials.access_token = Some(new_token.to_string());

                                // 更新过期时间
                                if let Some(expires_in) = data["expires_in"].as_i64() {
                                    let now = chrono::Utc::now();
                                    let expires_at = now + chrono::Duration::seconds(expires_in);
                                    self.credentials.expire = Some(expires_at.to_rfc3339());
                                    self.credentials.expiry_date =
                                        Some(expires_at.timestamp_millis());
                                    self.credentials.expires_in = Some(expires_in);
                                    self.credentials.timestamp = Some(now.timestamp_millis());
                                }

                                // 更新 refresh_token（如果返回了新的）
                                if let Some(new_refresh) = data["refresh_token"].as_str() {
                                    self.credentials.refresh_token = Some(new_refresh.to_string());
                                }

                                self.credentials.last_refresh =
                                    Some(chrono::Utc::now().to_rfc3339());

                                // 保存凭证
                                if let Err(e) = self.save_credentials().await {
                                    tracing::warn!("[Antigravity] 保存凭证失败: {}", e);
                                }

                                return Ok(new_token.to_string());
                            }
                            Err(e) => {
                                last_error = Some(TokenRefreshError::Unknown {
                                    message: format!("解析响应失败: {}", e),
                                });
                            }
                        }
                    } else {
                        // 请求失败
                        let status_code = status.as_u16();
                        let body = resp.text().await.unwrap_or_default();
                        let error = Self::classify_refresh_error(status_code, &body);

                        // invalid_grant 不重试
                        if error.requires_reauth() {
                            return Err(error);
                        }

                        // 可重试的错误
                        if error.is_retryable() {
                            last_error = Some(error);
                            retry_count += 1;
                            continue;
                        }

                        return Err(error);
                    }
                }
                Err(e) => {
                    // 网络错误，可重试
                    last_error = Some(TokenRefreshError::NetworkError {
                        message: e.to_string(),
                    });
                    retry_count += 1;
                    continue;
                }
            }

            retry_count += 1;
        }

        // 所有重试都失败
        Err(last_error.unwrap_or_else(|| TokenRefreshError::Unknown {
            message: "Token 刷新失败，已达到最大重试次数".to_string(),
        }))
    }

    pub async fn refresh_token(&mut self) -> Result<String, Box<dyn Error + Send + Sync>> {
        let refresh_token = self
            .credentials
            .refresh_token
            .as_ref()
            .ok_or("No refresh token available")?;

        let params = [
            ("client_id", OAUTH_CLIENT_ID),
            ("client_secret", OAUTH_CLIENT_SECRET),
            ("refresh_token", refresh_token.as_str()),
            ("grant_type", "refresh_token"),
        ];

        let resp = self
            .client
            .post("https://oauth2.googleapis.com/token")
            .form(&params)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Token refresh failed: {status} - {body}").into());
        }

        let data: serde_json::Value = resp.json().await?;

        let new_token = data["access_token"]
            .as_str()
            .ok_or("No access token in response")?;

        self.credentials.access_token = Some(new_token.to_string());

        // 更新过期时间（同时保存多种格式以兼容）
        if let Some(expires_in) = data["expires_in"].as_i64() {
            let now = chrono::Utc::now();
            let expires_at = now + chrono::Duration::seconds(expires_in);

            // RFC3339 格式
            self.credentials.expire = Some(expires_at.to_rfc3339());
            // 毫秒时间戳格式
            self.credentials.expiry_date = Some(expires_at.timestamp_millis());
            // antigravity2api-nodejs 格式
            self.credentials.expires_in = Some(expires_in);
            self.credentials.timestamp = Some(now.timestamp_millis());
        }

        // 如果返回了新的 refresh_token，也更新它
        if let Some(new_refresh) = data["refresh_token"].as_str() {
            self.credentials.refresh_token = Some(new_refresh.to_string());
        }

        // 更新最后刷新时间（RFC3339 格式）
        self.credentials.last_refresh = Some(chrono::Utc::now().to_rfc3339());

        // Save refreshed credentials
        self.save_credentials().await?;

        Ok(new_token.to_string())
    }

    /// 调用 Antigravity API
    async fn call_api_internal(
        &self,
        base_url: &str,
        method: &str,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, Box<dyn Error + Send + Sync>> {
        let token = self
            .credentials
            .access_token
            .as_ref()
            .ok_or("No access token")?;

        let url = format!("{}/{ANTIGRAVITY_API_VERSION}:{method}", base_url);

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {token}"))
            .header("Content-Type", "application/json")
            .header("User-Agent", "antigravity/1.11.5 windows/amd64")
            .json(body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("API call failed: {status} - {body}").into());
        }

        let data: serde_json::Value = resp.json().await?;
        Ok(data)
    }

    /// 调用 API，支持多环境降级
    pub async fn call_api(
        &self,
        method: &str,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, Box<dyn Error + Send + Sync>> {
        let mut last_error: Option<Box<dyn Error + Send + Sync>> = None;

        for base_url in &self.base_urls {
            match self.call_api_internal(base_url, method, body).await {
                Ok(data) => return Ok(data),
                Err(e) => {
                    tracing::warn!("[Antigravity] Failed on {}: {}", base_url, e);
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| "All Antigravity base URLs failed".into()))
    }

    /// 发现项目 ID
    pub async fn discover_project(&mut self) -> Result<String, Box<dyn Error + Send + Sync>> {
        if let Some(ref project_id) = self.project_id {
            return Ok(project_id.clone());
        }

        let body = serde_json::json!({
            "cloudaicompanionProject": "",
            "metadata": {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
                "duetProject": ""
            }
        });

        let resp = self.call_api("loadCodeAssist", &body).await?;

        if let Some(project) = resp["cloudaicompanionProject"].as_str() {
            if !project.is_empty() {
                self.project_id = Some(project.to_string());
                return Ok(project.to_string());
            }
        }

        // Need to onboard
        let onboard_body = serde_json::json!({
            "tierId": "free-tier",
            "cloudaicompanionProject": "",
            "metadata": {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
                "duetProject": ""
            }
        });

        let mut lro_resp = self.call_api("onboardUser", &onboard_body).await?;

        // Poll until done
        for _ in 0..30 {
            if lro_resp["done"].as_bool().unwrap_or(false) {
                break;
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
            lro_resp = self.call_api("onboardUser", &onboard_body).await?;
        }

        let project_id = lro_resp["response"]["cloudaicompanionProject"]["id"]
            .as_str()
            .unwrap_or("")
            .to_string();

        if project_id.is_empty() {
            // 生成一个随机项目 ID 作为后备
            let fallback = generate_project_id();
            self.project_id = Some(fallback.clone());
            return Ok(fallback);
        }

        self.project_id = Some(project_id.clone());
        Ok(project_id)
    }

    /// 获取可用模型列表
    pub async fn fetch_available_models(
        &mut self,
    ) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
        let body = serde_json::json!({});

        match self.call_api("fetchAvailableModels", &body).await {
            Ok(resp) => {
                if let Some(models) = resp["models"].as_object() {
                    self.available_models = models
                        .keys()
                        .filter_map(|name| {
                            let alias = model_name_to_alias(name);
                            if alias.is_empty() {
                                None
                            } else {
                                Some(alias.to_string())
                            }
                        })
                        .collect();
                }
            }
            Err(e) => {
                tracing::warn!(
                    "[Antigravity] Failed to fetch models: {}, using defaults",
                    e
                );
            }
        }

        Ok(self.available_models.clone())
    }

    /// 生成内容（非流式）
    pub async fn generate_content(
        &self,
        model: &str,
        request_body: &serde_json::Value,
    ) -> Result<serde_json::Value, Box<dyn Error + Send + Sync>> {
        let project_id = self.project_id.clone().unwrap_or_else(generate_project_id);
        let actual_model = alias_to_model_name(model);

        let payload = self.build_antigravity_request(actual_model, &project_id, request_body);

        let resp = self.call_api("generateContent", &payload).await?;

        // 转换为 Gemini 格式响应
        Ok(self.to_gemini_response(&resp))
    }

    /// 构建 Antigravity 请求
    ///
    /// 注意：此方法用于简单的非流式请求。
    /// 对于完整的 OpenAI 格式转换，请使用 `convert_openai_to_antigravity_with_context`。
    fn build_antigravity_request(
        &self,
        model: &str,
        project_id: &str,
        request_body: &serde_json::Value,
    ) -> serde_json::Value {
        let mut payload = request_body.clone();

        // 设置基本字段
        payload["model"] = serde_json::json!(model);
        payload["userAgent"] = serde_json::json!("antigravity");
        payload["project"] = serde_json::json!(project_id);
        payload["requestId"] = serde_json::json!(generate_request_id());

        // 确保 request 对象存在
        if payload.get("request").is_none() {
            payload["request"] = serde_json::json!({});
        }

        // 设置会话 ID
        payload["request"]["sessionId"] = serde_json::json!(generate_session_id());

        // 添加默认安全设置（如果不存在）
        if payload
            .get("request")
            .and_then(|r| r.get("safetySettings"))
            .is_none()
        {
            payload["request"]["safetySettings"] = serde_json::json!([
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"}
            ]);
        }

        payload
    }

    /// 转换为 Gemini 格式响应
    fn to_gemini_response(&self, antigravity_resp: &serde_json::Value) -> serde_json::Value {
        let mut response = serde_json::json!({});

        if let Some(candidates) = antigravity_resp.get("candidates") {
            response["candidates"] = candidates.clone();
        }

        if let Some(usage) = antigravity_resp.get("usageMetadata") {
            response["usageMetadata"] = usage.clone();
        }

        if let Some(feedback) = antigravity_resp.get("promptFeedback") {
            response["promptFeedback"] = feedback.clone();
        }

        response
    }

    /// 检查模型是否支持
    pub fn supports_model(&self, model: &str) -> bool {
        self.available_models.iter().any(|m| m == model)
    }
}

// ============================================================================
// OAuth 登录功能
// ============================================================================

/// OAuth 回调结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthCallbackResult {
    pub code: String,
    pub state: String,
}

/// OAuth 登录成功后的凭证信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntigravityOAuthResult {
    pub credentials: AntigravityCredentials,
    pub creds_file_path: String,
}

/// 生成 OAuth 授权 URL
pub fn generate_auth_url(port: u16, state: &str) -> String {
    let scopes = OAUTH_SCOPES.join(" ");
    let redirect_uri = format!("http://localhost:{}/oauth-callback", port);

    let params = [
        ("access_type", "offline"),
        ("client_id", OAUTH_CLIENT_ID),
        ("prompt", "consent"),
        ("redirect_uri", &redirect_uri),
        ("response_type", "code"),
        ("scope", &scopes),
        ("state", state),
    ];

    let query = params
        .iter()
        .map(|(k, v)| format!("{}={}", k, urlencoding::encode(v)))
        .collect::<Vec<_>>()
        .join("&");

    format!("https://accounts.google.com/o/oauth2/v2/auth?{}", query)
}

/// 用授权码交换 Token
pub async fn exchange_code_for_token(
    client: &Client,
    code: &str,
    redirect_uri: &str,
) -> Result<serde_json::Value, Box<dyn Error + Send + Sync>> {
    let params = [
        ("code", code),
        ("client_id", OAUTH_CLIENT_ID),
        ("client_secret", OAUTH_CLIENT_SECRET),
        ("redirect_uri", redirect_uri),
        ("grant_type", "authorization_code"),
    ];

    let resp = client
        .post("https://oauth2.googleapis.com/token")
        .form(&params)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Token 交换失败: {} - {}", status, body).into());
    }

    let data: serde_json::Value = resp.json().await?;
    Ok(data)
}

/// 获取用户邮箱
pub async fn fetch_user_email(
    client: &Client,
    access_token: &str,
) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
    let resp = client
        .get("https://www.googleapis.com/oauth2/v2/userinfo")
        .header("Authorization", format!("Bearer {}", access_token))
        .send()
        .await?;

    if resp.status().is_success() {
        let data: serde_json::Value = resp.json().await?;
        Ok(data["email"].as_str().map(|s| s.to_string()))
    } else {
        Ok(None)
    }
}

/// 获取项目 ID（验证账号资格）
/// 返回值说明：
/// - Ok(Some(FetchedProjectId::HasProject(id))) - 有资格，且有 projectId
/// - Ok(Some(FetchedProjectId::NoProject)) - 有资格，但 projectId 为空（需要生成随机 ID）
/// - Ok(None) - 无资格（字段不存在，即 undefined）
/// - Err(_) - 请求失败
#[derive(Debug, Clone)]
pub enum FetchedProjectId {
    /// 有 projectId
    HasProject(String),
    /// projectId 为空字符串（有资格但无 projectId）
    NoProject,
}

pub async fn fetch_project_id_for_oauth(
    client: &Client,
    access_token: &str,
) -> Result<Option<FetchedProjectId>, Box<dyn Error + Send + Sync>> {
    tracing::info!("[Antigravity OAuth] 正在获取 projectId...");

    let resp = client
        .post("https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal:loadCodeAssist")
        .header("Authorization", format!("Bearer {}", access_token))
        .header("User-Agent", "antigravity/1.11.9 windows/amd64")
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({ "metadata": { "ideType": "ANTIGRAVITY" } }))
        .send()
        .await?;

    let status = resp.status();
    tracing::info!("[Antigravity OAuth] loadCodeAssist 响应状态: {}", status);

    if status.is_success() {
        let body_text = resp.text().await?;
        tracing::info!("[Antigravity OAuth] loadCodeAssist 响应体: {}", body_text);

        let data: serde_json::Value = serde_json::from_str(&body_text)?;
        // 检查字段是否存在
        // - 如果字段不存在（undefined）-> None（无资格）
        // - 如果字段存在但为空字符串 -> Some(NoProject)（有资格但无 projectId）
        // - 如果字段存在且有值 -> Some(HasProject(id))（有资格且有 projectId）
        match data.get("cloudaicompanionProject") {
            None => {
                tracing::warn!("[Antigravity OAuth] cloudaicompanionProject 字段不存在");
                Ok(None) // 字段不存在，无资格
            }
            Some(value) => {
                if value.is_null() {
                    tracing::warn!("[Antigravity OAuth] cloudaicompanionProject 为 null");
                    Ok(None) // null 也视为无资格
                } else if let Some(s) = value.as_str() {
                    if s.is_empty() {
                        tracing::info!("[Antigravity OAuth] cloudaicompanionProject 为空字符串，有资格但无 projectId");
                        Ok(Some(FetchedProjectId::NoProject)) // 空字符串，有资格但无 projectId
                    } else {
                        tracing::info!("[Antigravity OAuth] 获取到 project_id: {}", s);
                        Ok(Some(FetchedProjectId::HasProject(s.to_string()))) // 有 projectId
                    }
                } else {
                    tracing::warn!(
                        "[Antigravity OAuth] cloudaicompanionProject 不是字符串类型: {:?}",
                        value
                    );
                    Ok(None) // 非字符串类型，视为无资格
                }
            }
        }
    } else {
        let body = resp.text().await.unwrap_or_default();
        tracing::error!(
            "[Antigravity OAuth] loadCodeAssist 请求失败: {} - {}",
            status,
            body
        );
        Err(format!("loadCodeAssist 请求失败: {} - {}", status, body).into())
    }
}

/// OAuth 成功页面 HTML
const OAUTH_SUCCESS_HTML: &str = r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>授权成功</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .container { text-align: center; background: white; padding: 40px 60px; border-radius: 16px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); }
        h1 { color: #22c55e; margin-bottom: 16px; }
        p { color: #666; margin-bottom: 8px; }
        .email { color: #333; font-weight: 500; }
    </style>
</head>
<body>
    <div class="container">
        <h1>✓ 授权成功</h1>
        <p>账号已添加到 ProxyCast</p>
        <p class="email">EMAIL_PLACEHOLDER</p>
        <p style="margin-top: 20px; color: #999;">可以关闭此页面</p>
    </div>
</body>
</html>"#;

/// OAuth 失败页面 HTML
const OAUTH_ERROR_HTML: &str = r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>授权失败</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .container { text-align: center; background: white; padding: 40px 60px; border-radius: 16px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); }
        h1 { color: #ef4444; margin-bottom: 16px; }
        p { color: #666; }
        .error { color: #ef4444; font-size: 14px; margin-top: 16px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>✗ 授权失败</h1>
        <p>ERROR_PLACEHOLDER</p>
        <p style="margin-top: 20px; color: #999;">请关闭此页面后重试</p>
    </div>
</body>
</html>"#;

/// OAuth 授权 URL 结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthAuthUrlResult {
    pub auth_url: String,
    pub port: u16,
    pub state: String,
}

/// 启动 OAuth 服务器并返回授权 URL（不打开浏览器）
/// 服务器会在后台等待回调，成功后返回凭证
pub async fn start_oauth_server_and_get_url(
    skip_project_id_fetch: bool,
) -> Result<
    (
        String,
        impl std::future::Future<Output = Result<AntigravityOAuthResult, Box<dyn Error + Send + Sync>>>,
    ),
    Box<dyn Error + Send + Sync>,
> {
    use axum::{extract::Query, response::Html, routing::get, Router};
    use std::collections::HashMap;
    use tokio::net::TcpListener;

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    // 生成随机 state
    let state = Uuid::new_v4().to_string();
    let state_clone = state.clone();

    // 创建 channel 用于接收回调结果
    let (tx, rx) = oneshot::channel::<Result<AntigravityOAuthResult, String>>();
    let tx = Arc::new(tokio::sync::Mutex::new(Some(tx)));

    // 绑定到随机端口
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();

    let redirect_uri = format!("http://localhost:{}/oauth-callback", port);
    let redirect_uri_clone = redirect_uri.clone();

    // 生成授权 URL
    let auth_url = generate_auth_url(port, &state);

    tracing::info!(
        "[Antigravity OAuth] 服务器启动在端口 {}, 授权 URL: {}",
        port,
        auth_url
    );

    // 构建路由
    let app = Router::new().route(
        "/oauth-callback",
        get(move |Query(params): Query<HashMap<String, String>>| {
            let tx = tx.clone();
            let client = client.clone();
            let state_expected = state_clone.clone();
            let redirect_uri = redirect_uri_clone.clone();

            async move {
                let code = params.get("code");
                let returned_state = params.get("state");
                let error = params.get("error");

                // 检查错误
                if let Some(err) = error {
                    let html = OAUTH_ERROR_HTML.replace("ERROR_PLACEHOLDER", err);
                    if let Some(sender) = tx.lock().await.take() {
                        let _ = sender.send(Err(format!("OAuth 错误: {}", err)));
                    }
                    return Html(html);
                }

                // 检查 state
                if returned_state.map(|s| s.as_str()) != Some(&state_expected) {
                    let html = OAUTH_ERROR_HTML.replace("ERROR_PLACEHOLDER", "State 验证失败");
                    if let Some(sender) = tx.lock().await.take() {
                        let _ = sender.send(Err("State 验证失败".to_string()));
                    }
                    return Html(html);
                }

                // 检查 code
                let code = match code {
                    Some(c) => c,
                    None => {
                        let html = OAUTH_ERROR_HTML.replace("ERROR_PLACEHOLDER", "未收到授权码");
                        if let Some(sender) = tx.lock().await.take() {
                            let _ = sender.send(Err("未收到授权码".to_string()));
                        }
                        return Html(html);
                    }
                };

                // 交换 Token
                let token_result = exchange_code_for_token(&client, code, &redirect_uri).await;
                let token_data = match token_result {
                    Ok(data) => data,
                    Err(e) => {
                        let html = OAUTH_ERROR_HTML.replace("ERROR_PLACEHOLDER", &e.to_string());
                        if let Some(sender) = tx.lock().await.take() {
                            let _ = sender.send(Err(e.to_string()));
                        }
                        return Html(html);
                    }
                };

                let access_token = token_data["access_token"].as_str().unwrap_or_default();
                let refresh_token = token_data["refresh_token"].as_str().map(|s| s.to_string());
                let expires_in = token_data["expires_in"].as_i64();

                // 获取用户邮箱
                let email = fetch_user_email(&client, access_token).await.ok().flatten();

                // 获取项目 ID
                let project_id = if skip_project_id_fetch {
                    tracing::info!("[Antigravity OAuth] 跳过 projectId 获取，使用随机生成的 ID");
                    Some(generate_project_id())
                } else {
                    match fetch_project_id_for_oauth(&client, access_token).await {
                        Ok(Some(FetchedProjectId::HasProject(pid))) => Some(pid),
                        Ok(Some(FetchedProjectId::NoProject)) => {
                            tracing::info!("[Antigravity OAuth] projectId 为空，使用随机生成的 ID");
                            Some(generate_project_id())
                        }
                        Ok(None) => {
                            tracing::warn!("[Antigravity OAuth] 无法获取 projectId（字段不存在），使用随机生成的 ID");
                            Some(generate_project_id())
                        }
                        Err(e) => {
                            tracing::warn!("[Antigravity OAuth] 获取 projectId 失败: {}, 使用随机 ID", e);
                            Some(generate_project_id())
                        }
                    }
                };

                // 构建凭证
                let now = chrono::Utc::now();
                let credentials = AntigravityCredentials {
                    access_token: Some(access_token.to_string()),
                    refresh_token,
                    token_type: Some("Bearer".to_string()),
                    expiry_date: expires_in.map(|e| (now + chrono::Duration::seconds(e)).timestamp_millis()),
                    expire: expires_in.map(|e| (now + chrono::Duration::seconds(e)).to_rfc3339()),
                    scope: Some(OAUTH_SCOPES.join(" ")),
                    last_refresh: Some(now.to_rfc3339()),
                    cred_type: "antigravity".to_string(),
                    expires_in,
                    timestamp: Some(now.timestamp_millis()),
                    enable: Some(true),
                    project_id: project_id,
                    email: email.clone(),
                };

                // 保存凭证到应用数据目录
                let creds_dir = dirs::data_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join("proxycast")
                    .join("credentials")
                    .join("antigravity");

                if let Err(e) = std::fs::create_dir_all(&creds_dir) {
                    let html = OAUTH_ERROR_HTML.replace("ERROR_PLACEHOLDER", &format!("创建目录失败: {}", e));
                    if let Some(sender) = tx.lock().await.take() {
                        let _ = sender.send(Err(e.to_string()));
                    }
                    return Html(html);
                }

                // 使用 UUID 作为文件名
                let file_name = format!("{}.json", Uuid::new_v4());
                let creds_path = creds_dir.join(&file_name);

                let creds_json = match serde_json::to_string_pretty(&credentials) {
                    Ok(json) => json,
                    Err(e) => {
                        let html = OAUTH_ERROR_HTML.replace("ERROR_PLACEHOLDER", &format!("序列化失败: {}", e));
                        if let Some(sender) = tx.lock().await.take() {
                            let _ = sender.send(Err(e.to_string()));
                        }
                        return Html(html);
                    }
                };

                if let Err(e) = std::fs::write(&creds_path, &creds_json) {
                    let html = OAUTH_ERROR_HTML.replace("ERROR_PLACEHOLDER", &format!("保存凭证失败: {}", e));
                    if let Some(sender) = tx.lock().await.take() {
                        let _ = sender.send(Err(e.to_string()));
                    }
                    return Html(html);
                }

                let creds_path_str = creds_path.to_string_lossy().to_string();
                tracing::info!("[Antigravity OAuth] 凭证已保存到: {}", creds_path_str);

                // 发送成功结果
                let result = AntigravityOAuthResult {
                    credentials,
                    creds_file_path: creds_path_str,
                };

                if let Some(sender) = tx.lock().await.take() {
                    let _ = sender.send(Ok(result));
                }

                let email_display = email.unwrap_or_else(|| "未知邮箱".to_string());
                let html = OAUTH_SUCCESS_HTML.replace("EMAIL_PLACEHOLDER", &email_display);
                Html(html)
            }
        }),
    );

    // 创建等待回调的 Future
    let wait_future = async move {
        // 启动服务器
        let server = axum::serve(listener, app);

        // 同时运行服务器和等待回调结果
        tokio::select! {
            result = async {
                tokio::time::timeout(
                    std::time::Duration::from_secs(300),
                    rx
                ).await
            } => {
                match result {
                    Ok(Ok(Ok(r))) => Ok(r),
                    Ok(Ok(Err(e))) => Err(e.into()),
                    Ok(Err(_)) => Err("OAuth 回调通道关闭".into()),
                    Err(_) => Err("OAuth 登录超时（5分钟）".into()),
                }
            }
            server_result = server => {
                match server_result {
                    Ok(_) => Err("服务器意外关闭".into()),
                    Err(e) => Err(format!("服务器错误: {}", e).into()),
                }
            }
        }
    };

    Ok((auth_url, wait_future))
}

/// 启动 OAuth 登录流程（使用指定端口）
/// 用于配合 get_oauth_auth_url 使用
pub async fn start_oauth_login_with_port(
    port: u16,
    state: String,
    skip_project_id_fetch: bool,
) -> Result<AntigravityOAuthResult, Box<dyn Error + Send + Sync>> {
    use axum::{extract::Query, response::Html, routing::get, Router};
    use std::collections::HashMap;
    use tokio::net::TcpListener;

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    let state_clone = state.clone();

    // 创建 channel 用于接收回调结果
    let (tx, rx) = oneshot::channel::<Result<AntigravityOAuthResult, String>>();
    let tx = Arc::new(tokio::sync::Mutex::new(Some(tx)));

    // 绑定到指定端口
    let listener = TcpListener::bind(format!("127.0.0.1:{}", port)).await?;

    let redirect_uri = format!("http://localhost:{}/oauth-callback", port);
    let redirect_uri_clone = redirect_uri.clone();

    // 构建路由
    let app = Router::new().route(
        "/oauth-callback",
        get(move |Query(params): Query<HashMap<String, String>>| {
            let tx = tx.clone();
            let client = client.clone();
            let state_expected = state_clone.clone();
            let redirect_uri = redirect_uri_clone.clone();

            async move {
                let code = params.get("code");
                let returned_state = params.get("state");
                let error = params.get("error");

                // 检查错误
                if let Some(err) = error {
                    let html = OAUTH_ERROR_HTML.replace("ERROR_PLACEHOLDER", err);
                    if let Some(sender) = tx.lock().await.take() {
                        let _ = sender.send(Err(format!("OAuth 错误: {}", err)));
                    }
                    return Html(html);
                }

                // 检查 state
                if returned_state.map(|s| s.as_str()) != Some(&state_expected) {
                    let html = OAUTH_ERROR_HTML.replace("ERROR_PLACEHOLDER", "State 验证失败");
                    if let Some(sender) = tx.lock().await.take() {
                        let _ = sender.send(Err("State 验证失败".to_string()));
                    }
                    return Html(html);
                }

                // 检查 code
                let code = match code {
                    Some(c) => c,
                    None => {
                        let html = OAUTH_ERROR_HTML.replace("ERROR_PLACEHOLDER", "未收到授权码");
                        if let Some(sender) = tx.lock().await.take() {
                            let _ = sender.send(Err("未收到授权码".to_string()));
                        }
                        return Html(html);
                    }
                };

                // 交换 Token
                let token_result = exchange_code_for_token(&client, code, &redirect_uri).await;
                let token_data = match token_result {
                    Ok(data) => data,
                    Err(e) => {
                        let html = OAUTH_ERROR_HTML.replace("ERROR_PLACEHOLDER", &e.to_string());
                        if let Some(sender) = tx.lock().await.take() {
                            let _ = sender.send(Err(e.to_string()));
                        }
                        return Html(html);
                    }
                };

                let access_token = token_data["access_token"].as_str().unwrap_or_default();
                let refresh_token = token_data["refresh_token"].as_str().map(|s| s.to_string());
                let expires_in = token_data["expires_in"].as_i64();

                // 获取用户邮箱
                let email = fetch_user_email(&client, access_token).await.ok().flatten();

                // 获取项目 ID
                let project_id = if skip_project_id_fetch {
                    tracing::info!("[Antigravity OAuth] 跳过 projectId 获取，使用随机生成的 ID");
                    Some(generate_project_id())
                } else {
                    match fetch_project_id_for_oauth(&client, access_token).await {
                        Ok(Some(FetchedProjectId::HasProject(pid))) => Some(pid),
                        Ok(Some(FetchedProjectId::NoProject)) => {
                            tracing::info!("[Antigravity OAuth] projectId 为空，使用随机生成的 ID");
                            Some(generate_project_id())
                        }
                        Ok(None) => {
                            tracing::warn!("[Antigravity OAuth] 无法获取 projectId（字段不存在），使用随机生成的 ID");
                            Some(generate_project_id())
                        }
                        Err(e) => {
                            tracing::warn!("[Antigravity OAuth] 获取 projectId 失败: {}, 使用随机 ID", e);
                            Some(generate_project_id())
                        }
                    }
                };

                // 构建凭证
                let now = chrono::Utc::now();
                let credentials = AntigravityCredentials {
                    access_token: Some(access_token.to_string()),
                    refresh_token,
                    token_type: Some("Bearer".to_string()),
                    expiry_date: expires_in.map(|e| (now + chrono::Duration::seconds(e)).timestamp_millis()),
                    expire: expires_in.map(|e| (now + chrono::Duration::seconds(e)).to_rfc3339()),
                    scope: Some(OAUTH_SCOPES.join(" ")),
                    last_refresh: Some(now.to_rfc3339()),
                    cred_type: "antigravity".to_string(),
                    expires_in,
                    timestamp: Some(now.timestamp_millis()),
                    enable: Some(true),
                    project_id: project_id,
                    email: email.clone(),
                };

                // 保存凭证到应用数据目录
                let creds_dir = dirs::data_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join("proxycast")
                    .join("credentials")
                    .join("antigravity");

                if let Err(e) = std::fs::create_dir_all(&creds_dir) {
                    let html = OAUTH_ERROR_HTML.replace("ERROR_PLACEHOLDER", &format!("创建目录失败: {}", e));
                    if let Some(sender) = tx.lock().await.take() {
                        let _ = sender.send(Err(e.to_string()));
                    }
                    return Html(html);
                }

                // 使用 UUID 作为文件名
                let file_name = format!("{}.json", Uuid::new_v4());
                let creds_path = creds_dir.join(&file_name);

                let creds_json = match serde_json::to_string_pretty(&credentials) {
                    Ok(json) => json,
                    Err(e) => {
                        let html = OAUTH_ERROR_HTML.replace("ERROR_PLACEHOLDER", &format!("序列化失败: {}", e));
                        if let Some(sender) = tx.lock().await.take() {
                            let _ = sender.send(Err(e.to_string()));
                        }
                        return Html(html);
                    }
                };

                if let Err(e) = std::fs::write(&creds_path, &creds_json) {
                    let html = OAUTH_ERROR_HTML.replace("ERROR_PLACEHOLDER", &format!("保存凭证失败: {}", e));
                    if let Some(sender) = tx.lock().await.take() {
                        let _ = sender.send(Err(e.to_string()));
                    }
                    return Html(html);
                }

                let creds_path_str = creds_path.to_string_lossy().to_string();
                tracing::info!("[Antigravity OAuth] 凭证已保存到: {}", creds_path_str);

                // 发送成功结果
                let result = AntigravityOAuthResult {
                    credentials,
                    creds_file_path: creds_path_str,
                };

                if let Some(sender) = tx.lock().await.take() {
                    let _ = sender.send(Ok(result));
                }

                let email_display = email.unwrap_or_else(|| "未知邮箱".to_string());
                let html = OAUTH_SUCCESS_HTML.replace("EMAIL_PLACEHOLDER", &email_display);
                Html(html)
            }
        }),
    );

    // 启动服务器
    let server = axum::serve(listener, app);

    // 同时运行服务器和等待回调结果
    tokio::select! {
        result = async {
            tokio::time::timeout(
                std::time::Duration::from_secs(300),
                rx
            ).await
        } => {
            match result {
                Ok(Ok(Ok(r))) => Ok(r),
                Ok(Ok(Err(e))) => Err(e.into()),
                Ok(Err(_)) => Err("OAuth 回调通道关闭".into()),
                Err(_) => Err("OAuth 登录超时（5分钟）".into()),
            }
        }
        server_result = server => {
            match server_result {
                Ok(_) => Err("服务器意外关闭".into()),
                Err(e) => Err(format!("服务器错误: {}", e).into()),
            }
        }
    }
}

/// 启动 OAuth 登录流程
/// 返回 (auth_url, credentials_file_path)
pub async fn start_oauth_login(
    skip_project_id_fetch: bool,
) -> Result<AntigravityOAuthResult, Box<dyn Error + Send + Sync>> {
    use axum::{extract::Query, response::Html, routing::get, Router};
    use std::collections::HashMap;
    use tokio::net::TcpListener;

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    // 生成随机 state
    let state = Uuid::new_v4().to_string();
    let state_clone = state.clone();

    // 创建 channel 用于接收回调结果
    let (tx, rx) = oneshot::channel::<Result<AntigravityOAuthResult, String>>();
    let tx = Arc::new(tokio::sync::Mutex::new(Some(tx)));

    // 绑定到随机端口
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();

    let redirect_uri = format!("http://localhost:{}/oauth-callback", port);
    let redirect_uri_clone = redirect_uri.clone();

    // 构建路由
    let app = Router::new().route(
        "/oauth-callback",
        get(move |Query(params): Query<HashMap<String, String>>| {
            let tx = tx.clone();
            let client = client.clone();
            let state_expected = state_clone.clone();
            let redirect_uri = redirect_uri_clone.clone();

            async move {
                let code = params.get("code");
                let returned_state = params.get("state");
                let error = params.get("error");

                // 检查错误
                if let Some(err) = error {
                    let html = OAUTH_ERROR_HTML.replace("ERROR_PLACEHOLDER", err);
                    if let Some(sender) = tx.lock().await.take() {
                        let _ = sender.send(Err(format!("OAuth 错误: {}", err)));
                    }
                    return Html(html);
                }

                // 检查 state
                if returned_state.map(|s| s.as_str()) != Some(&state_expected) {
                    let html = OAUTH_ERROR_HTML.replace("ERROR_PLACEHOLDER", "State 验证失败");
                    if let Some(sender) = tx.lock().await.take() {
                        let _ = sender.send(Err("State 验证失败".to_string()));
                    }
                    return Html(html);
                }

                // 检查 code
                let code = match code {
                    Some(c) => c,
                    None => {
                        let html = OAUTH_ERROR_HTML.replace("ERROR_PLACEHOLDER", "未收到授权码");
                        if let Some(sender) = tx.lock().await.take() {
                            let _ = sender.send(Err("未收到授权码".to_string()));
                        }
                        return Html(html);
                    }
                };

                // 交换 Token
                let token_result = exchange_code_for_token(&client, code, &redirect_uri).await;
                let token_data = match token_result {
                    Ok(data) => data,
                    Err(e) => {
                        let html = OAUTH_ERROR_HTML.replace("ERROR_PLACEHOLDER", &e.to_string());
                        if let Some(sender) = tx.lock().await.take() {
                            let _ = sender.send(Err(e.to_string()));
                        }
                        return Html(html);
                    }
                };

                let access_token = token_data["access_token"].as_str().unwrap_or_default();
                let refresh_token = token_data["refresh_token"].as_str().map(|s| s.to_string());
                let expires_in = token_data["expires_in"].as_i64();

                // 获取用户邮箱
                let email = fetch_user_email(&client, access_token).await.ok().flatten();

                // 获取项目 ID
                // 参考 antigravity2api-nodejs 的逻辑：
                // - projectId === undefined -> 无资格（但我们改为使用随机 ID，因为很多账号都没有 projectId）
                // - projectId === "" -> 有资格但无 projectId，使用随机生成的
                // - projectId 有值 -> 有资格且有 projectId
                let project_id = if skip_project_id_fetch {
                    tracing::info!("[Antigravity OAuth] 跳过 projectId 获取，使用随机生成的 ID");
                    Some(generate_project_id())
                } else {
                    match fetch_project_id_for_oauth(&client, access_token).await {
                        Ok(Some(FetchedProjectId::HasProject(pid))) => {
                            // 有资格且有 projectId
                            Some(pid)
                        }
                        Ok(Some(FetchedProjectId::NoProject)) => {
                            // 有资格但 projectId 为空，使用随机生成的
                            tracing::info!("[Antigravity OAuth] projectId 为空，使用随机生成的 ID");
                            Some(generate_project_id())
                        }
                        Ok(None) => {
                            // 字段不存在，也使用随机 ID（很多账号都是这种情况）
                            tracing::warn!("[Antigravity OAuth] 无法获取 projectId（字段不存在），使用随机生成的 ID");
                            Some(generate_project_id())
                        }
                        Err(e) => {
                            tracing::warn!("[Antigravity OAuth] 获取 projectId 失败: {}, 使用随机 ID", e);
                            Some(generate_project_id())
                        }
                    }
                };

                // 构建凭证
                let now = chrono::Utc::now();
                let credentials = AntigravityCredentials {
                    access_token: Some(access_token.to_string()),
                    refresh_token,
                    token_type: Some("Bearer".to_string()),
                    expiry_date: expires_in.map(|e| (now + chrono::Duration::seconds(e)).timestamp_millis()),
                    expire: expires_in.map(|e| (now + chrono::Duration::seconds(e)).to_rfc3339()),
                    scope: Some(OAUTH_SCOPES.join(" ")),
                    last_refresh: Some(now.to_rfc3339()),
                    cred_type: "antigravity".to_string(),
                    expires_in,
                    timestamp: Some(now.timestamp_millis()),
                    enable: Some(true),
                    project_id: project_id,
                    email: email.clone(),
                };

                // 保存凭证到应用数据目录
                let creds_dir = dirs::data_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join("proxycast")
                    .join("credentials")
                    .join("antigravity");

                if let Err(e) = std::fs::create_dir_all(&creds_dir) {
                    let html = OAUTH_ERROR_HTML.replace("ERROR_PLACEHOLDER", &format!("创建目录失败: {}", e));
                    if let Some(sender) = tx.lock().await.take() {
                        let _ = sender.send(Err(e.to_string()));
                    }
                    return Html(html);
                }

                // 使用 UUID 作为文件名
                let file_name = format!("{}.json", Uuid::new_v4());
                let creds_path = creds_dir.join(&file_name);

                let creds_json = match serde_json::to_string_pretty(&credentials) {
                    Ok(json) => json,
                    Err(e) => {
                        let html = OAUTH_ERROR_HTML.replace("ERROR_PLACEHOLDER", &format!("序列化失败: {}", e));
                        if let Some(sender) = tx.lock().await.take() {
                            let _ = sender.send(Err(e.to_string()));
                        }
                        return Html(html);
                    }
                };

                if let Err(e) = std::fs::write(&creds_path, &creds_json) {
                    let html = OAUTH_ERROR_HTML.replace("ERROR_PLACEHOLDER", &format!("保存凭证失败: {}", e));
                    if let Some(sender) = tx.lock().await.take() {
                        let _ = sender.send(Err(e.to_string()));
                    }
                    return Html(html);
                }

                let creds_path_str = creds_path.to_string_lossy().to_string();
                tracing::info!("[Antigravity OAuth] 凭证已保存到: {}", creds_path_str);

                // 发送成功结果
                let result = AntigravityOAuthResult {
                    credentials,
                    creds_file_path: creds_path_str,
                };

                if let Some(sender) = tx.lock().await.take() {
                    let _ = sender.send(Ok(result));
                }

                let email_display = email.unwrap_or_else(|| "未知邮箱".to_string());
                let html = OAUTH_SUCCESS_HTML.replace("EMAIL_PLACEHOLDER", &email_display);
                Html(html)
            }
        }),
    );

    // 生成授权 URL
    let auth_url = generate_auth_url(port, &state);

    // 打开浏览器
    tracing::info!("[Antigravity OAuth] 打开浏览器进行授权: {}", auth_url);
    if let Err(e) = open::that(&auth_url) {
        tracing::warn!("[Antigravity OAuth] 无法自动打开浏览器: {}", e);
    }

    // 启动服务器
    let server = axum::serve(listener, app);

    // 同时运行服务器和等待回调结果
    tokio::select! {
        // 等待回调结果（带超时）
        result = async {
            tokio::time::timeout(
                std::time::Duration::from_secs(300),
                rx
            ).await
        } => {
            match result {
                Ok(Ok(Ok(r))) => Ok(r),
                Ok(Ok(Err(e))) => Err(e.into()),
                Ok(Err(_)) => Err("OAuth 回调通道关闭".into()),
                Err(_) => Err("OAuth 登录超时（5分钟）".into()),
            }
        }
        // 服务器运行（不会主动结束，除非出错）
        server_result = server => {
            match server_result {
                Ok(_) => Err("服务器意外关闭".into()),
                Err(e) => Err(format!("服务器错误: {}", e).into()),
            }
        }
    }
}

// ============================================================================
// CredentialProvider Trait 实现
// ============================================================================

#[async_trait]
impl CredentialProvider for AntigravityProvider {
    async fn load_credentials_from_path(&mut self, path: &str) -> ProviderResult<()> {
        AntigravityProvider::load_credentials_from_path(self, path).await
    }

    async fn save_credentials(&self) -> ProviderResult<()> {
        AntigravityProvider::save_credentials(self).await
    }

    fn is_token_valid(&self) -> bool {
        AntigravityProvider::is_token_valid(self)
    }

    fn is_token_expiring_soon(&self) -> bool {
        AntigravityProvider::is_token_expiring_soon(self)
    }

    async fn refresh_token(&mut self) -> ProviderResult<String> {
        AntigravityProvider::refresh_token(self).await
    }

    fn get_access_token(&self) -> Option<&str> {
        self.credentials.access_token.as_deref()
    }

    fn provider_type(&self) -> &'static str {
        "antigravity"
    }
}

// ============================================================================
// StreamingProvider Trait 实现
// ============================================================================

use crate::converter::openai_to_antigravity::convert_openai_to_antigravity_with_context;
use crate::models::openai::ChatCompletionRequest;
use crate::providers::ProviderError;
use crate::streaming::traits::{
    reqwest_stream_to_stream_response, StreamFormat, StreamResponse, StreamingProvider,
};

#[async_trait]
impl StreamingProvider for AntigravityProvider {
    /// 发起流式 API 调用
    ///
    /// 使用 reqwest 的 bytes_stream 返回字节流，支持真正的端到端流式传输。
    /// Antigravity 使用 Gemini 流式格式。
    ///
    /// # 需求覆盖
    /// - 需求 1.4: AntigravityProvider 流式支持
    async fn call_api_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<StreamResponse, ProviderError> {
        tracing::info!("[ANTIGRAVITY_STREAM] ========== call_api_stream 开始 ==========");

        let token = self
            .credentials
            .access_token
            .as_ref()
            .ok_or_else(|| ProviderError::AuthenticationError("No access token".to_string()))?;

        tracing::info!("[ANTIGRAVITY_STREAM] Token 长度: {} 字符", token.len());

        let project_id = self.project_id.clone().unwrap_or_else(generate_project_id);
        let actual_model = alias_to_model_name(&request.model);

        tracing::info!(
            "[ANTIGRAVITY_STREAM] project_id={}, request.model={}, actual_model={}",
            project_id,
            request.model,
            actual_model
        );

        // 使用统一的转换函数构建请求体
        let payload = convert_openai_to_antigravity_with_context(request, &project_id);

        tracing::info!(
            "[ANTIGRAVITY_STREAM] 请求体 (完整): {}",
            serde_json::to_string_pretty(&payload).unwrap_or_default()
        );

        // 尝试多个 base URL
        let mut last_error: Option<ProviderError> = None;

        for base_url in &self.base_urls {
            let url = format!(
                "{}/{ANTIGRAVITY_API_VERSION}:streamGenerateContent",
                base_url
            );

            eprintln!("[ANTIGRAVITY_STREAM] ========== 发起 HTTP 请求 ==========");
            eprintln!("[ANTIGRAVITY_STREAM] URL: {}", url);
            eprintln!("[ANTIGRAVITY_STREAM] Model: {}", actual_model);
            eprintln!(
                "[ANTIGRAVITY_STREAM] Token 前20字符: {}...",
                &token[..20.min(token.len())]
            );
            tracing::info!(
                "[ANTIGRAVITY_STREAM] ========== 发起 HTTP 请求 ==========\n  URL: {}\n  Model: {}\n  Method: POST",
                url,
                actual_model
            );

            let result = self
                .client
                .post(&url)
                .header("Authorization", format!("Bearer {}", token))
                .header("Content-Type", "application/json")
                .header("Accept", "text/event-stream")
                .header("User-Agent", "antigravity/1.11.5 windows/amd64")
                .json(&payload)
                .send()
                .await;

            match result {
                Ok(resp) => {
                    let status = resp.status();
                    eprintln!("[ANTIGRAVITY_STREAM] HTTP 响应状态: {}", status);
                    tracing::info!("[ANTIGRAVITY_STREAM] HTTP 响应状态: {}", status);

                    if status.is_success() {
                        eprintln!("[ANTIGRAVITY_STREAM] ✓ 流式响应成功建立");
                        tracing::info!("[ANTIGRAVITY_STREAM] ✓ 流式响应成功建立，返回流");
                        return Ok(reqwest_stream_to_stream_response(resp));
                    } else {
                        let body = resp.text().await.unwrap_or_default();
                        eprintln!(
                            "[ANTIGRAVITY_STREAM] ✗ 请求失败\n  Base URL: {}\n  Status: {}\n  Body: {}",
                            base_url,
                            status,
                            &body[..body.len().min(500)]
                        );
                        tracing::error!(
                            "[ANTIGRAVITY_STREAM] ✗ 请求失败\n  Base URL: {}\n  Status: {}\n  Body: {}",
                            base_url,
                            status,
                            body
                        );
                        last_error = Some(ProviderError::from_http_status(status.as_u16(), &body));
                    }
                }
                Err(e) => {
                    eprintln!(
                        "[ANTIGRAVITY_STREAM] ✗ 连接失败\n  Base URL: {}\n  Error: {}",
                        base_url, e
                    );
                    tracing::error!(
                        "[ANTIGRAVITY_STREAM] ✗ 连接失败\n  Base URL: {}\n  Error: {}",
                        base_url,
                        e
                    );
                    last_error = Some(ProviderError::from_reqwest_error(&e));
                }
            }
        }

        tracing::error!("[ANTIGRAVITY_STREAM] 所有 base URL 都失败了");
        Err(last_error.unwrap_or_else(|| {
            ProviderError::NetworkError("All Antigravity base URLs failed".to_string())
        }))
    }

    fn supports_streaming(&self) -> bool {
        self.credentials.access_token.is_some() && self.credentials.enable != Some(false)
    }

    fn provider_name(&self) -> &'static str {
        "AntigravityProvider"
    }

    fn stream_format(&self) -> StreamFormat {
        StreamFormat::GeminiStream
    }
}

// ==================== 测试模块 ====================

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // 辅助函数：检查是否为 Valid 状态
    fn is_valid(result: &TokenValidationResult) -> bool {
        matches!(result, TokenValidationResult::Valid { .. })
    }

    // 辅助函数：检查是否为 ExpiringSoon 状态
    fn is_expiring_soon(result: &TokenValidationResult) -> bool {
        matches!(result, TokenValidationResult::ExpiringSoon { .. })
    }

    // 辅助函数：检查是否为 Expired 状态
    fn is_expired(result: &TokenValidationResult) -> bool {
        matches!(result, TokenValidationResult::Expired)
    }

    // 辅助函数：检查是否为 Invalid 状态
    fn is_invalid(result: &TokenValidationResult) -> bool {
        matches!(result, TokenValidationResult::Invalid { .. })
    }

    // ==================== Property 1: Token 过期时间解析正确性 ====================
    // Feature: antigravity-token-refresh, Property 1: Token 过期时间解析正确性
    // Validates: Requirements 1.1, 1.3

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Property 1: 对于任何有效的过期时间（RFC3339 格式），validate_token() 应正确判断状态
        #[test]
        fn prop_validate_token_rfc3339_format(
            expires_in_secs in -3600i64..7200i64, // -1小时到2小时
        ) {
            let now = chrono::Utc::now();
            let expires_at = now + chrono::Duration::seconds(expires_in_secs);

            let mut provider = AntigravityProvider::new();
            provider.credentials.access_token = Some("test_token".to_string());
            provider.credentials.refresh_token = Some("test_refresh".to_string());
            provider.credentials.expire = Some(expires_at.to_rfc3339());

            let result = provider.validate_token();

            if expires_in_secs <= 0 {
                prop_assert!(is_expired(&result), "Expected Expired for expires_in_secs={}", expires_in_secs);
            } else if expires_in_secs <= TOKEN_EXPIRING_SOON_THRESHOLD {
                prop_assert!(is_expiring_soon(&result), "Expected ExpiringSoon for expires_in_secs={}", expires_in_secs);
            } else {
                prop_assert!(is_valid(&result), "Expected Valid for expires_in_secs={}", expires_in_secs);
            }
        }

        /// Property 1: 对于任何有效的过期时间（毫秒时间戳格式），validate_token() 应正确判断状态
        #[test]
        fn prop_validate_token_timestamp_format(
            expires_in_secs in -3600i64..7200i64,
        ) {
            let now = chrono::Utc::now();
            let expires_at = now + chrono::Duration::seconds(expires_in_secs);

            let mut provider = AntigravityProvider::new();
            provider.credentials.access_token = Some("test_token".to_string());
            provider.credentials.refresh_token = Some("test_refresh".to_string());
            provider.credentials.expiry_date = Some(expires_at.timestamp_millis());

            let result = provider.validate_token();

            if expires_in_secs <= 0 {
                prop_assert!(is_expired(&result), "Expected Expired for expires_in_secs={}", expires_in_secs);
            } else if expires_in_secs <= TOKEN_EXPIRING_SOON_THRESHOLD {
                prop_assert!(is_expiring_soon(&result), "Expected ExpiringSoon for expires_in_secs={}", expires_in_secs);
            } else {
                prop_assert!(is_valid(&result), "Expected Valid for expires_in_secs={}", expires_in_secs);
            }
        }

        /// Property 1: 对于任何有效的过期时间（timestamp + expires_in 格式），validate_token() 应正确判断状态
        #[test]
        fn prop_validate_token_expires_in_format(
            expires_in_secs in 1i64..7200i64, // 只测试正数，因为这个格式不支持负数
        ) {
            let now = chrono::Utc::now();

            let mut provider = AntigravityProvider::new();
            provider.credentials.access_token = Some("test_token".to_string());
            provider.credentials.refresh_token = Some("test_refresh".to_string());
            provider.credentials.timestamp = Some(now.timestamp_millis());
            provider.credentials.expires_in = Some(expires_in_secs);

            let result = provider.validate_token();

            // 由于时间精度问题，允许 1 秒的误差
            if expires_in_secs <= 1 {
                prop_assert!(is_expired(&result) || is_expiring_soon(&result), "Expected Expired or ExpiringSoon for expires_in_secs={}", expires_in_secs);
            } else if expires_in_secs <= TOKEN_EXPIRING_SOON_THRESHOLD {
                prop_assert!(is_expiring_soon(&result), "Expected ExpiringSoon for expires_in_secs={}", expires_in_secs);
            } else {
                prop_assert!(is_valid(&result), "Expected Valid for expires_in_secs={}", expires_in_secs);
            }
        }
    }

    // ==================== Property 2: 空 Token 检测 ====================
    // Feature: antigravity-token-refresh, Property 2: 空 Token 检测
    // Validates: Requirements 1.2

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Property 2: 对于任何空或仅包含空白字符的 token，validate_token() 应返回 Invalid
        #[test]
        fn prop_validate_token_empty_detection(
            whitespace in "[ \t\n\r]*", // 生成各种空白字符组合
        ) {
            let mut provider = AntigravityProvider::new();
            provider.credentials.access_token = Some(whitespace);
            provider.credentials.refresh_token = Some("test_refresh".to_string());

            let result = provider.validate_token();

            prop_assert!(is_invalid(&result), "Expected Invalid for empty/whitespace token");
        }
    }

    /// Property 2: None token 应返回 Invalid
    #[test]
    fn test_validate_token_none() {
        let mut provider = AntigravityProvider::new();
        provider.credentials.access_token = None;
        provider.credentials.refresh_token = Some("test_refresh".to_string());

        let result = provider.validate_token();

        assert!(
            matches!(result, TokenValidationResult::Invalid { reason } if reason.contains("缺失"))
        );
    }

    /// Property 2: 缺少 refresh_token 应返回 Invalid
    #[test]
    fn test_validate_token_no_refresh_token() {
        let mut provider = AntigravityProvider::new();
        provider.credentials.access_token = Some("test_token".to_string());
        provider.credentials.refresh_token = None;

        let result = provider.validate_token();

        assert!(
            matches!(result, TokenValidationResult::Invalid { reason } if reason.contains("refresh_token"))
        );
    }

    /// Property 2: 禁用的凭证应返回 Invalid
    #[test]
    fn test_validate_token_disabled() {
        let mut provider = AntigravityProvider::new();
        provider.credentials.access_token = Some("test_token".to_string());
        provider.credentials.refresh_token = Some("test_refresh".to_string());
        provider.credentials.enable = Some(false);

        let result = provider.validate_token();

        assert!(
            matches!(result, TokenValidationResult::Invalid { reason } if reason.contains("禁用"))
        );
    }

    // ==================== TokenRefreshError 测试 ====================

    #[test]
    fn test_classify_refresh_error_invalid_grant() {
        let error =
            AntigravityProvider::classify_refresh_error(400, r#"{"error": "invalid_grant"}"#);
        assert!(matches!(error, TokenRefreshError::InvalidGrant { .. }));
        assert!(error.requires_reauth());
        assert!(!error.is_retryable());
    }

    #[test]
    fn test_classify_refresh_error_server_error() {
        let error = AntigravityProvider::classify_refresh_error(500, "Internal Server Error");
        assert!(matches!(error, TokenRefreshError::ServerError { .. }));
        assert!(!error.requires_reauth());
        assert!(error.is_retryable());
    }

    #[test]
    fn test_classify_refresh_error_unknown() {
        let error = AntigravityProvider::classify_refresh_error(403, "Forbidden");
        assert!(matches!(error, TokenRefreshError::Unknown { .. }));
        assert!(!error.requires_reauth());
        assert!(!error.is_retryable());
    }

    // ==================== TokenValidationResult 方法测试 ====================

    #[test]
    fn test_token_validation_result_needs_refresh() {
        assert!(!TokenValidationResult::Valid {
            expires_in_secs: 3600
        }
        .needs_refresh());
        assert!(TokenValidationResult::ExpiringSoon {
            expires_in_secs: 300
        }
        .needs_refresh());
        assert!(TokenValidationResult::Expired.needs_refresh());
        assert!(TokenValidationResult::Invalid {
            reason: "test".to_string()
        }
        .needs_refresh());
    }

    #[test]
    fn test_token_validation_result_is_usable() {
        assert!(TokenValidationResult::Valid {
            expires_in_secs: 3600
        }
        .is_usable());
        assert!(TokenValidationResult::ExpiringSoon {
            expires_in_secs: 300
        }
        .is_usable());
        assert!(!TokenValidationResult::Expired.is_usable());
        assert!(!TokenValidationResult::Invalid {
            reason: "test".to_string()
        }
        .is_usable());
    }

    // ==================== Property 5: 重试次数限制 ====================
    // Feature: antigravity-token-refresh, Property 5: 重试次数限制
    // Validates: Requirements 2.2

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Property 5: 对于任何 HTTP 状态码，错误分类应正确识别可重试错误
        #[test]
        fn prop_classify_error_retryable(
            status in 100u16..600u16,
        ) {
            let error = AntigravityProvider::classify_refresh_error(status, "test error");

            // 5xx 错误应该是可重试的
            if status >= 500 {
                prop_assert!(error.is_retryable(), "5xx errors should be retryable");
            }

            // 400 + invalid_grant 不应该重试
            if status == 400 {
                let invalid_grant_error = AntigravityProvider::classify_refresh_error(400, "invalid_grant");
                prop_assert!(!invalid_grant_error.is_retryable(), "invalid_grant should not be retryable");
                prop_assert!(invalid_grant_error.requires_reauth(), "invalid_grant should require reauth");
            }
        }

        /// Property 5: 对于任何错误类型，user_message 应返回非空字符串
        #[test]
        fn prop_error_user_message_not_empty(
            status in 100u16..600u16,
            body in ".*",
        ) {
            let error = AntigravityProvider::classify_refresh_error(status, &body);
            let message = error.user_message();
            prop_assert!(!message.is_empty(), "User message should not be empty");
        }
    }

    /// 测试 TokenRefreshError 的 Display 实现
    #[test]
    fn test_token_refresh_error_display() {
        let errors = vec![
            TokenRefreshError::InvalidGrant {
                message: "test".to_string(),
            },
            TokenRefreshError::NetworkError {
                message: "test".to_string(),
            },
            TokenRefreshError::ServerError {
                message: "test".to_string(),
            },
            TokenRefreshError::Unknown {
                message: "test".to_string(),
            },
        ];

        for error in errors {
            let display = format!("{}", error);
            assert!(!display.is_empty());
        }
    }

    /// 测试缺少 refresh_token 时 refresh_token_with_retry 应返回 InvalidGrant 错误
    #[tokio::test]
    async fn test_refresh_token_with_retry_no_refresh_token() {
        let mut provider = AntigravityProvider::new();
        provider.credentials.access_token = Some("test_token".to_string());
        provider.credentials.refresh_token = None;

        let result = provider.refresh_token_with_retry(3).await;

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.requires_reauth());
    }
}
