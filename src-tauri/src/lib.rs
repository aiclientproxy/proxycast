mod commands;
mod config;
mod converter;
pub mod credential;
mod database;
pub mod injection;
mod logger;
pub mod middleware;
mod models;
pub mod plugin;
pub mod processor;
mod providers;
pub mod proxy;
pub mod resilience;
pub mod router;
mod server;
mod services;
pub mod telemetry;
pub mod tray;
pub mod websocket;

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tauri::{Manager, Runtime};
use tokio::sync::RwLock;

use commands::provider_pool_cmd::{CredentialSyncServiceState, ProviderPoolServiceState};
use commands::resilience_cmd::ResilienceConfigState;
use commands::router_cmd::RouterConfigState;
use commands::skill_cmd::SkillServiceState;
use services::provider_pool_service::ProviderPoolService;
use services::skill_service::SkillService;
use services::token_cache_service::TokenCacheService;
use tray::{
    calculate_icon_status, CredentialHealth, TrayIconStatus, TrayManager, TrayStateSnapshot,
};

/// TokenCacheService 状态封装
pub struct TokenCacheServiceState(pub Arc<TokenCacheService>);

/// TrayManager 状态封装
///
/// 用于在 Tauri 状态管理中存储托盘管理器
pub struct TrayManagerState<R: Runtime>(pub Arc<tokio::sync::RwLock<Option<TrayManager<R>>>>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    Kiro,
    Gemini,
    Qwen,
    #[serde(rename = "openai")]
    OpenAI,
    Claude,
    Antigravity,
    Vertex,
    /// Gemini API Key (multi-account load balancing)
    #[serde(rename = "gemini_api_key")]
    GeminiApiKey,
}

impl std::fmt::Display for ProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderType::Kiro => write!(f, "kiro"),
            ProviderType::Gemini => write!(f, "gemini"),
            ProviderType::Qwen => write!(f, "qwen"),
            ProviderType::OpenAI => write!(f, "openai"),
            ProviderType::Claude => write!(f, "claude"),
            ProviderType::Antigravity => write!(f, "antigravity"),
            ProviderType::Vertex => write!(f, "vertex"),
            ProviderType::GeminiApiKey => write!(f, "gemini_api_key"),
        }
    }
}

impl std::str::FromStr for ProviderType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "kiro" => Ok(ProviderType::Kiro),
            "gemini" => Ok(ProviderType::Gemini),
            "qwen" => Ok(ProviderType::Qwen),
            "openai" => Ok(ProviderType::OpenAI),
            "claude" => Ok(ProviderType::Claude),
            "antigravity" => Ok(ProviderType::Antigravity),
            "vertex" => Ok(ProviderType::Vertex),
            "gemini_api_key" => Ok(ProviderType::GeminiApiKey),
            _ => Err(format!("Invalid provider: {s}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_type_from_str() {
        assert_eq!("kiro".parse::<ProviderType>().unwrap(), ProviderType::Kiro);
        assert_eq!(
            "gemini".parse::<ProviderType>().unwrap(),
            ProviderType::Gemini
        );
        assert_eq!("qwen".parse::<ProviderType>().unwrap(), ProviderType::Qwen);
        assert_eq!(
            "openai".parse::<ProviderType>().unwrap(),
            ProviderType::OpenAI
        );
        assert_eq!(
            "claude".parse::<ProviderType>().unwrap(),
            ProviderType::Claude
        );
        assert_eq!(
            "vertex".parse::<ProviderType>().unwrap(),
            ProviderType::Vertex
        );
        assert_eq!(
            "gemini_api_key".parse::<ProviderType>().unwrap(),
            ProviderType::GeminiApiKey
        );

        // 测试大小写不敏感
        assert_eq!("KIRO".parse::<ProviderType>().unwrap(), ProviderType::Kiro);
        assert_eq!(
            "Gemini".parse::<ProviderType>().unwrap(),
            ProviderType::Gemini
        );
        assert_eq!(
            "VERTEX".parse::<ProviderType>().unwrap(),
            ProviderType::Vertex
        );

        // 测试无效的 provider
        assert!("invalid".parse::<ProviderType>().is_err());
    }

    #[test]
    fn test_provider_type_display() {
        assert_eq!(ProviderType::Kiro.to_string(), "kiro");
        assert_eq!(ProviderType::Gemini.to_string(), "gemini");
        assert_eq!(ProviderType::Qwen.to_string(), "qwen");
        assert_eq!(ProviderType::OpenAI.to_string(), "openai");
        assert_eq!(ProviderType::Claude.to_string(), "claude");
        assert_eq!(ProviderType::Vertex.to_string(), "vertex");
        assert_eq!(ProviderType::GeminiApiKey.to_string(), "gemini_api_key");
    }

    #[test]
    fn test_provider_type_serde() {
        // 测试序列化
        assert_eq!(
            serde_json::to_string(&ProviderType::Kiro).unwrap(),
            "\"kiro\""
        );
        assert_eq!(
            serde_json::to_string(&ProviderType::OpenAI).unwrap(),
            "\"openai\""
        );

        // 测试反序列化
        assert_eq!(
            serde_json::from_str::<ProviderType>("\"kiro\"").unwrap(),
            ProviderType::Kiro
        );
        assert_eq!(
            serde_json::from_str::<ProviderType>("\"openai\"").unwrap(),
            ProviderType::OpenAI
        );
    }
}

pub type AppState = Arc<RwLock<server::ServerState>>;
pub type LogState = Arc<RwLock<logger::LogStore>>;

#[tauri::command]
async fn start_server(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
    db: tauri::State<'_, database::DbConnection>,
    pool_service: tauri::State<'_, ProviderPoolServiceState>,
    token_cache: tauri::State<'_, TokenCacheServiceState>,
) -> Result<String, String> {
    let mut s = state.write().await;
    logs.write().await.add("info", "Starting server...");
    s.start(
        logs.inner().clone(),
        pool_service.0.clone(),
        token_cache.0.clone(),
        Some(db.inner().clone()),
    )
    .await
    .map_err(|e| e.to_string())?;
    logs.write().await.add(
        "info",
        &format!(
            "Server started on {}:{}",
            s.config.server.host, s.config.server.port
        ),
    );
    Ok("Server started".to_string())
}

#[tauri::command]
async fn stop_server(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
) -> Result<String, String> {
    let mut s = state.write().await;
    s.stop().await;
    logs.write().await.add("info", "Server stopped");
    Ok("Server stopped".to_string())
}

#[tauri::command]
async fn get_server_status(
    state: tauri::State<'_, AppState>,
) -> Result<server::ServerStatus, String> {
    let s = state.read().await;
    Ok(s.status())
}

#[tauri::command]
async fn get_config(state: tauri::State<'_, AppState>) -> Result<config::Config, String> {
    let s = state.read().await;
    Ok(s.config.clone())
}

#[tauri::command]
async fn save_config(
    state: tauri::State<'_, AppState>,
    config: config::Config,
) -> Result<(), String> {
    let mut s = state.write().await;
    s.config = config.clone();
    config::save_config(&config).map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_default_provider(state: tauri::State<'_, AppState>) -> Result<String, String> {
    let s = state.read().await;
    Ok(s.config.default_provider.clone())
}

#[tauri::command]
async fn set_default_provider(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
    provider: String,
) -> Result<String, String> {
    // 使用枚举验证 provider
    let provider_type: ProviderType = provider.parse().map_err(|e: String| e)?;

    let mut s = state.write().await;
    s.config.default_provider = provider.clone();

    // 同时更新运行中服务器的 default_provider_ref
    {
        let mut dp = s.default_provider_ref.write().await;
        *dp = provider.clone();
    }

    config::save_config(&s.config).map_err(|e| e.to_string())?;
    logs.write()
        .await
        .add("info", &format!("默认 Provider 已切换为: {provider_type}"));
    Ok(provider)
}

#[tauri::command]
async fn refresh_kiro_token(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
) -> Result<String, String> {
    let mut s = state.write().await;
    logs.write().await.add("info", "Refreshing Kiro token...");
    let result = s
        .kiro_provider
        .refresh_token()
        .await
        .map_err(|e| e.to_string());
    match &result {
        Ok(_) => logs
            .write()
            .await
            .add("info", "Token refreshed successfully"),
        Err(e) => logs
            .write()
            .await
            .add("error", &format!("Token refresh failed: {e}")),
    }
    result
}

#[tauri::command]
async fn reload_credentials(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
) -> Result<String, String> {
    let mut s = state.write().await;
    logs.write().await.add("info", "Reloading credentials...");
    s.kiro_provider
        .load_credentials()
        .await
        .map_err(|e| e.to_string())?;
    logs.write().await.add("info", "Credentials reloaded");
    Ok("Credentials reloaded".to_string())
}

#[derive(serde::Serialize)]
struct KiroCredentialStatus {
    loaded: bool,
    has_access_token: bool,
    has_refresh_token: bool,
    region: Option<String>,
    auth_method: Option<String>,
    expires_at: Option<String>,
    creds_path: String,
}

#[tauri::command]
async fn get_kiro_credentials(
    state: tauri::State<'_, AppState>,
) -> Result<KiroCredentialStatus, String> {
    let s = state.read().await;
    let creds = &s.kiro_provider.credentials;
    let path = providers::kiro::KiroProvider::default_creds_path();

    Ok(KiroCredentialStatus {
        loaded: creds.access_token.is_some() || creds.refresh_token.is_some(),
        has_access_token: creds.access_token.is_some(),
        has_refresh_token: creds.refresh_token.is_some(),
        region: creds.region.clone(),
        auth_method: creds.auth_method.clone(),
        expires_at: creds.expires_at.clone(),
        creds_path: path.to_string_lossy().to_string(),
    })
}

#[derive(serde::Serialize)]
struct EnvVariable {
    key: String,
    value: String,
    masked: String,
}

#[tauri::command]
async fn get_env_variables(state: tauri::State<'_, AppState>) -> Result<Vec<EnvVariable>, String> {
    let s = state.read().await;
    let creds = &s.kiro_provider.credentials;
    let mut vars = Vec::new();

    if let Some(token) = &creds.access_token {
        vars.push(EnvVariable {
            key: "KIRO_ACCESS_TOKEN".to_string(),
            value: token.clone(),
            masked: mask_token(token),
        });
    }
    if let Some(token) = &creds.refresh_token {
        vars.push(EnvVariable {
            key: "KIRO_REFRESH_TOKEN".to_string(),
            value: token.clone(),
            masked: mask_token(token),
        });
    }
    if let Some(id) = &creds.client_id {
        vars.push(EnvVariable {
            key: "KIRO_CLIENT_ID".to_string(),
            value: id.clone(),
            masked: mask_token(id),
        });
    }
    if let Some(secret) = &creds.client_secret {
        vars.push(EnvVariable {
            key: "KIRO_CLIENT_SECRET".to_string(),
            value: secret.clone(),
            masked: mask_token(secret),
        });
    }
    if let Some(arn) = &creds.profile_arn {
        vars.push(EnvVariable {
            key: "KIRO_PROFILE_ARN".to_string(),
            value: arn.clone(),
            masked: arn.clone(),
        });
    }
    if let Some(region) = &creds.region {
        vars.push(EnvVariable {
            key: "KIRO_REGION".to_string(),
            value: region.clone(),
            masked: region.clone(),
        });
    }
    if let Some(method) = &creds.auth_method {
        vars.push(EnvVariable {
            key: "KIRO_AUTH_METHOD".to_string(),
            value: method.clone(),
            masked: method.clone(),
        });
    }

    Ok(vars)
}

fn mask_token(token: &str) -> String {
    let chars: Vec<char> = token.chars().collect();
    if chars.len() <= 12 {
        "****".to_string()
    } else {
        let prefix: String = chars[..6].iter().collect();
        let suffix: String = chars[chars.len() - 4..].iter().collect();
        format!("{prefix}****{suffix}")
    }
}

#[tauri::command]
async fn get_token_file_hash() -> Result<String, String> {
    let path = providers::kiro::KiroProvider::default_creds_path();
    if !tokio::fs::try_exists(&path).await.unwrap_or(false) {
        return Ok("".to_string());
    }

    let content = tokio::fs::read(&path).await.map_err(|e| e.to_string())?;
    let hash = format!("{:x}", md5::compute(&content));
    Ok(hash)
}

/// 检查凭证文件变化并自动重新加载（带日志记录）
#[tauri::command]
async fn check_and_reload_credentials(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
    last_hash: String,
) -> Result<CheckResult, String> {
    let path = providers::kiro::KiroProvider::default_creds_path();

    if !tokio::fs::try_exists(&path).await.unwrap_or(false) {
        return Ok(CheckResult {
            changed: false,
            new_hash: "".to_string(),
            reloaded: false,
        });
    }

    let content = tokio::fs::read(&path).await.map_err(|e| e.to_string())?;
    let new_hash = format!("{:x}", md5::compute(&content));

    if !last_hash.is_empty() && new_hash != last_hash {
        logs.write()
            .await
            .add("info", "[自动检测] 凭证文件已变化，正在重新加载...");

        let mut s = state.write().await;
        match s.kiro_provider.load_credentials().await {
            Ok(_) => {
                logs.write()
                    .await
                    .add("info", "[自动检测] 凭证重新加载成功");
                Ok(CheckResult {
                    changed: true,
                    new_hash,
                    reloaded: true,
                })
            }
            Err(e) => {
                logs.write()
                    .await
                    .add("error", &format!("[自动检测] 凭证重新加载失败: {e}"));
                Ok(CheckResult {
                    changed: true,
                    new_hash,
                    reloaded: false,
                })
            }
        }
    } else {
        Ok(CheckResult {
            changed: false,
            new_hash,
            reloaded: false,
        })
    }
}

#[derive(serde::Serialize)]
struct CheckResult {
    changed: bool,
    new_hash: String,
    reloaded: bool,
}

// ============ Gemini Provider Commands ============

#[derive(serde::Serialize)]
struct GeminiCredentialStatus {
    loaded: bool,
    has_access_token: bool,
    has_refresh_token: bool,
    expiry_date: Option<i64>,
    is_valid: bool,
    creds_path: String,
}

#[tauri::command]
async fn get_gemini_credentials(
    state: tauri::State<'_, AppState>,
) -> Result<GeminiCredentialStatus, String> {
    let s = state.read().await;
    let creds = &s.gemini_provider.credentials;
    let path = providers::gemini::GeminiProvider::default_creds_path();

    Ok(GeminiCredentialStatus {
        loaded: creds.access_token.is_some() || creds.refresh_token.is_some(),
        has_access_token: creds.access_token.is_some(),
        has_refresh_token: creds.refresh_token.is_some(),
        expiry_date: creds.expiry_date,
        is_valid: s.gemini_provider.is_token_valid(),
        creds_path: path.to_string_lossy().to_string(),
    })
}

#[tauri::command]
async fn reload_gemini_credentials(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
) -> Result<String, String> {
    let mut s = state.write().await;
    logs.write().await.add("info", "[Gemini] 正在加载凭证...");
    s.gemini_provider
        .load_credentials()
        .await
        .map_err(|e| e.to_string())?;
    logs.write().await.add("info", "[Gemini] 凭证加载成功");
    Ok("Gemini credentials reloaded".to_string())
}

#[tauri::command]
async fn refresh_gemini_token(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
) -> Result<String, String> {
    let mut s = state.write().await;
    logs.write().await.add("info", "[Gemini] 正在刷新 Token...");
    let result = s
        .gemini_provider
        .refresh_token()
        .await
        .map_err(|e| e.to_string());
    match &result {
        Ok(_) => logs.write().await.add("info", "[Gemini] Token 刷新成功"),
        Err(e) => logs
            .write()
            .await
            .add("error", &format!("[Gemini] Token 刷新失败: {e}")),
    }
    result
}

#[tauri::command]
async fn get_gemini_env_variables(
    state: tauri::State<'_, AppState>,
) -> Result<Vec<EnvVariable>, String> {
    let s = state.read().await;
    let creds = &s.gemini_provider.credentials;
    let mut vars = Vec::new();

    if let Some(token) = &creds.access_token {
        vars.push(EnvVariable {
            key: "GEMINI_ACCESS_TOKEN".to_string(),
            value: token.clone(),
            masked: mask_token(token),
        });
    }
    if let Some(token) = &creds.refresh_token {
        vars.push(EnvVariable {
            key: "GEMINI_REFRESH_TOKEN".to_string(),
            value: token.clone(),
            masked: mask_token(token),
        });
    }
    if let Some(expiry) = creds.expiry_date {
        let expiry_str = expiry.to_string();
        vars.push(EnvVariable {
            key: "GEMINI_EXPIRY_DATE".to_string(),
            value: expiry_str.clone(),
            masked: expiry_str,
        });
    }

    Ok(vars)
}

#[tauri::command]
async fn get_gemini_token_file_hash() -> Result<String, String> {
    let path = providers::gemini::GeminiProvider::default_creds_path();
    if !tokio::fs::try_exists(&path).await.unwrap_or(false) {
        return Ok("".to_string());
    }

    let content = tokio::fs::read(&path).await.map_err(|e| e.to_string())?;
    let hash = format!("{:x}", md5::compute(&content));
    Ok(hash)
}

#[tauri::command]
async fn check_and_reload_gemini_credentials(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
    last_hash: String,
) -> Result<CheckResult, String> {
    let path = providers::gemini::GeminiProvider::default_creds_path();

    if !tokio::fs::try_exists(&path).await.unwrap_or(false) {
        return Ok(CheckResult {
            changed: false,
            new_hash: "".to_string(),
            reloaded: false,
        });
    }

    let content = tokio::fs::read(&path).await.map_err(|e| e.to_string())?;
    let new_hash = format!("{:x}", md5::compute(&content));

    if !last_hash.is_empty() && new_hash != last_hash {
        logs.write()
            .await
            .add("info", "[Gemini][自动检测] 凭证文件已变化，正在重新加载...");

        let mut s = state.write().await;
        match s.gemini_provider.load_credentials().await {
            Ok(_) => {
                logs.write()
                    .await
                    .add("info", "[Gemini][自动检测] 凭证重新加载成功");
                Ok(CheckResult {
                    changed: true,
                    new_hash,
                    reloaded: true,
                })
            }
            Err(e) => {
                logs.write().await.add(
                    "error",
                    &format!("[Gemini][自动检测] 凭证重新加载失败: {e}"),
                );
                Ok(CheckResult {
                    changed: true,
                    new_hash,
                    reloaded: false,
                })
            }
        }
    } else {
        Ok(CheckResult {
            changed: false,
            new_hash,
            reloaded: false,
        })
    }
}

// ============ Qwen Provider Commands ============

#[derive(serde::Serialize)]
struct QwenCredentialStatus {
    loaded: bool,
    has_access_token: bool,
    has_refresh_token: bool,
    expiry_date: Option<i64>,
    is_valid: bool,
    creds_path: String,
}

#[tauri::command]
async fn get_qwen_credentials(
    state: tauri::State<'_, AppState>,
) -> Result<QwenCredentialStatus, String> {
    let s = state.read().await;
    let creds = &s.qwen_provider.credentials;
    let path = providers::qwen::QwenProvider::default_creds_path();

    Ok(QwenCredentialStatus {
        loaded: creds.access_token.is_some() || creds.refresh_token.is_some(),
        has_access_token: creds.access_token.is_some(),
        has_refresh_token: creds.refresh_token.is_some(),
        expiry_date: creds.expiry_date,
        is_valid: s.qwen_provider.is_token_valid(),
        creds_path: path.to_string_lossy().to_string(),
    })
}

#[tauri::command]
async fn reload_qwen_credentials(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
) -> Result<String, String> {
    let mut s = state.write().await;
    logs.write().await.add("info", "[Qwen] 正在加载凭证...");
    s.qwen_provider
        .load_credentials()
        .await
        .map_err(|e| e.to_string())?;
    logs.write().await.add("info", "[Qwen] 凭证加载成功");
    Ok("Qwen credentials reloaded".to_string())
}

#[tauri::command]
async fn refresh_qwen_token(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
) -> Result<String, String> {
    let mut s = state.write().await;
    logs.write().await.add("info", "[Qwen] 正在刷新 Token...");
    let result = s
        .qwen_provider
        .refresh_token()
        .await
        .map_err(|e| e.to_string());
    match &result {
        Ok(_) => logs.write().await.add("info", "[Qwen] Token 刷新成功"),
        Err(e) => logs
            .write()
            .await
            .add("error", &format!("[Qwen] Token 刷新失败: {e}")),
    }
    result
}

#[tauri::command]
async fn get_qwen_env_variables(
    state: tauri::State<'_, AppState>,
) -> Result<Vec<EnvVariable>, String> {
    let s = state.read().await;
    let creds = &s.qwen_provider.credentials;
    let mut vars = Vec::new();

    if let Some(token) = &creds.access_token {
        vars.push(EnvVariable {
            key: "QWEN_ACCESS_TOKEN".to_string(),
            value: token.clone(),
            masked: mask_token(token),
        });
    }
    if let Some(token) = &creds.refresh_token {
        vars.push(EnvVariable {
            key: "QWEN_REFRESH_TOKEN".to_string(),
            value: token.clone(),
            masked: mask_token(token),
        });
    }
    if let Some(url) = &creds.resource_url {
        vars.push(EnvVariable {
            key: "QWEN_RESOURCE_URL".to_string(),
            value: url.clone(),
            masked: url.clone(),
        });
    }
    if let Some(expiry) = creds.expiry_date {
        let expiry_str = expiry.to_string();
        vars.push(EnvVariable {
            key: "QWEN_EXPIRY_DATE".to_string(),
            value: expiry_str.clone(),
            masked: expiry_str,
        });
    }

    Ok(vars)
}

#[tauri::command]
async fn get_qwen_token_file_hash() -> Result<String, String> {
    let path = providers::qwen::QwenProvider::default_creds_path();
    if !tokio::fs::try_exists(&path).await.unwrap_or(false) {
        return Ok("".to_string());
    }

    let content = tokio::fs::read(&path).await.map_err(|e| e.to_string())?;
    let hash = format!("{:x}", md5::compute(&content));
    Ok(hash)
}

#[tauri::command]
async fn check_and_reload_qwen_credentials(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
    last_hash: String,
) -> Result<CheckResult, String> {
    let path = providers::qwen::QwenProvider::default_creds_path();

    if !tokio::fs::try_exists(&path).await.unwrap_or(false) {
        return Ok(CheckResult {
            changed: false,
            new_hash: "".to_string(),
            reloaded: false,
        });
    }

    let content = tokio::fs::read(&path).await.map_err(|e| e.to_string())?;
    let new_hash = format!("{:x}", md5::compute(&content));

    if !last_hash.is_empty() && new_hash != last_hash {
        logs.write()
            .await
            .add("info", "[Qwen][自动检测] 凭证文件已变化，正在重新加载...");

        let mut s = state.write().await;
        match s.qwen_provider.load_credentials().await {
            Ok(_) => {
                logs.write()
                    .await
                    .add("info", "[Qwen][自动检测] 凭证重新加载成功");
                Ok(CheckResult {
                    changed: true,
                    new_hash,
                    reloaded: true,
                })
            }
            Err(e) => {
                logs.write()
                    .await
                    .add("error", &format!("[Qwen][自动检测] 凭证重新加载失败: {e}"));
                Ok(CheckResult {
                    changed: true,
                    new_hash,
                    reloaded: false,
                })
            }
        }
    } else {
        Ok(CheckResult {
            changed: false,
            new_hash,
            reloaded: false,
        })
    }
}

// ============ OpenAI Custom Provider Commands ============

#[derive(serde::Serialize, serde::Deserialize)]
struct OpenAICustomStatus {
    enabled: bool,
    has_api_key: bool,
    base_url: String,
}

#[tauri::command]
async fn get_openai_custom_status(
    state: tauri::State<'_, AppState>,
) -> Result<OpenAICustomStatus, String> {
    let s = state.read().await;
    let config = &s.openai_custom_provider.config;
    Ok(OpenAICustomStatus {
        enabled: config.enabled,
        has_api_key: config.api_key.is_some(),
        base_url: s.openai_custom_provider.get_base_url(),
    })
}

#[tauri::command]
async fn set_openai_custom_config(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
    api_key: Option<String>,
    base_url: Option<String>,
    enabled: bool,
) -> Result<String, String> {
    let mut s = state.write().await;
    s.openai_custom_provider.config.api_key = api_key;
    s.openai_custom_provider.config.base_url = base_url;
    s.openai_custom_provider.config.enabled = enabled;
    logs.write().await.add(
        "info",
        &format!("[OpenAI Custom] 配置已更新, enabled={enabled}"),
    );
    Ok("OpenAI Custom config updated".to_string())
}

// ============ Claude Custom Provider Commands ============

#[derive(serde::Serialize, serde::Deserialize)]
struct ClaudeCustomStatus {
    enabled: bool,
    has_api_key: bool,
    base_url: String,
}

#[tauri::command]
async fn get_claude_custom_status(
    state: tauri::State<'_, AppState>,
) -> Result<ClaudeCustomStatus, String> {
    let s = state.read().await;
    let config = &s.claude_custom_provider.config;
    Ok(ClaudeCustomStatus {
        enabled: config.enabled,
        has_api_key: config.api_key.is_some(),
        base_url: s.claude_custom_provider.get_base_url(),
    })
}

#[tauri::command]
async fn set_claude_custom_config(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
    api_key: Option<String>,
    base_url: Option<String>,
    enabled: bool,
) -> Result<String, String> {
    let mut s = state.write().await;
    s.claude_custom_provider.config.api_key = api_key;
    s.claude_custom_provider.config.base_url = base_url;
    s.claude_custom_provider.config.enabled = enabled;
    logs.write().await.add(
        "info",
        &format!("[Claude Custom] 配置已更新, enabled={enabled}"),
    );
    Ok("Claude Custom config updated".to_string())
}

#[tauri::command]
async fn get_logs(logs: tauri::State<'_, LogState>) -> Result<Vec<logger::LogEntry>, String> {
    Ok(logs.read().await.get_logs())
}

#[tauri::command]
async fn clear_logs(logs: tauri::State<'_, LogState>) -> Result<(), String> {
    logs.write().await.clear();
    Ok(())
}

#[derive(serde::Serialize)]
struct TestResult {
    success: bool,
    status: u16,
    body: String,
    time_ms: u64,
}

#[derive(serde::Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    owned_by: String,
}

// ============ API Compatibility Check ============

#[derive(serde::Serialize)]
struct ApiCheckResult {
    model: String,
    available: bool,
    status: u16,
    error_type: Option<String>,
    error_message: Option<String>,
    time_ms: u64,
}

#[derive(serde::Serialize)]
struct ApiCompatibilityResult {
    provider: String,
    overall_status: String,
    checked_at: String,
    results: Vec<ApiCheckResult>,
    warnings: Vec<String>,
}

#[tauri::command]
async fn check_api_compatibility(
    state: tauri::State<'_, AppState>,
    logs: tauri::State<'_, LogState>,
    provider: String,
) -> Result<ApiCompatibilityResult, String> {
    // 使用枚举验证 provider
    let provider_type: ProviderType = provider.parse().map_err(|e: String| e)?;

    logs.write().await.add(
        "info",
        &format!("[API检测] 开始检测 {provider_type} API 兼容性 (Claude Code 功能测试)..."),
    );

    let s = state.read().await;
    let mut results: Vec<ApiCheckResult> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    // Claude Code 需要的测试项目
    let test_cases: Vec<(&str, &str)> = match provider_type {
        ProviderType::Kiro => vec![
            ("claude-sonnet-4-5", "basic"),     // 基础对话
            ("claude-sonnet-4-5", "tool_call"), // Tool Calls 支持
        ],
        ProviderType::Gemini => vec![
            ("gemini-2.5-flash", "basic"),
            ("gemini-2.5-flash", "tool_call"),
        ],
        ProviderType::Qwen => vec![
            ("qwen3-coder-plus", "basic"),
            ("qwen3-coder-plus", "tool_call"),
        ],
        ProviderType::Antigravity => vec![
            ("gemini-3-pro-preview", "basic"),
            ("gemini-3-pro-preview", "tool_call"),
        ],
        ProviderType::Vertex => vec![
            ("gemini-2.0-flash", "basic"),
            ("gemini-2.0-flash", "tool_call"),
        ],
        ProviderType::GeminiApiKey => vec![
            ("gemini-2.5-flash", "basic"),
            ("gemini-2.5-flash", "tool_call"),
        ],
        ProviderType::OpenAI | ProviderType::Claude => vec![],
    };

    for (model, test_type) in test_cases {
        let start = std::time::Instant::now();
        let test_name = format!("{model} ({test_type})");

        // 根据测试类型构建不同的请求
        let test_request = match test_type {
            "tool_call" => {
                // 测试 Tool Calls - Claude Code 核心功能
                crate::models::openai::ChatCompletionRequest {
                    model: model.to_string(),
                    messages: vec![crate::models::openai::ChatMessage {
                        role: "user".to_string(),
                        content: Some(crate::models::openai::MessageContent::Text(
                            "What is 2+2? Use the calculator tool to compute this.".to_string(),
                        )),
                        tool_calls: None,
                        tool_call_id: None,
                    }],
                    temperature: None,
                    max_tokens: Some(100),
                    stream: false,
                    tools: Some(vec![crate::models::openai::Tool {
                        tool_type: "function".to_string(),
                        function: crate::models::openai::FunctionDef {
                            name: "calculator".to_string(),
                            description: Some("Perform basic arithmetic calculations".to_string()),
                            parameters: Some(serde_json::json!({
                                "type": "object",
                                "properties": {
                                    "expression": {
                                        "type": "string",
                                        "description": "The math expression to evaluate"
                                    }
                                },
                                "required": ["expression"]
                            })),
                        },
                    }]),
                    tool_choice: None,
                }
            }
            _ => {
                // 基础对话测试
                crate::models::openai::ChatCompletionRequest {
                    model: model.to_string(),
                    messages: vec![crate::models::openai::ChatMessage {
                        role: "user".to_string(),
                        content: Some(crate::models::openai::MessageContent::Text(
                            "Say 'OK' only.".to_string(),
                        )),
                        tool_calls: None,
                        tool_call_id: None,
                    }],
                    temperature: None,
                    max_tokens: Some(10),
                    stream: false,
                    tools: None,
                    tool_choice: None,
                }
            }
        };

        let result = match provider_type {
            ProviderType::Kiro => s.kiro_provider.call_api(&test_request).await,
            ProviderType::Gemini => {
                // Gemini 暂时不支持直接 API 检测，返回未实现错误
                Err("Gemini API compatibility check not yet implemented".into())
            }
            ProviderType::Qwen => {
                // Qwen 暂时不支持直接 API 检测，返回未实现错误
                Err("Qwen API compatibility check not yet implemented".into())
            }
            _ => Err("Provider not supported for direct API check".into()),
        };

        let time_ms = start.elapsed().as_millis() as u64;

        match result {
            Ok(resp) => {
                let status = resp.status().as_u16();
                let body = resp.text().await.unwrap_or_default();

                let (available, error_type, error_message) = if (200..300).contains(&status) {
                    // 对于 tool_call 测试，额外检查响应是否包含 tool use
                    if test_type == "tool_call" {
                        let has_tool_use =
                            body.contains("\"name\"") && body.contains("\"toolUseId\"");
                        if !has_tool_use {
                            warnings.push(format!(
                                "{test_name}: 响应未包含 tool_use，Claude Code 可能无法正常工作"
                            ));
                        }
                    }
                    (true, None, None)
                } else {
                    let err_type = match status {
                        401 => {
                            warnings.push(format!("{test_name} 返回 401: Token 可能已过期或无效"));
                            Some("AUTH_ERROR".to_string())
                        }
                        403 => {
                            warnings.push(format!(
                                "{test_name} 返回 403: 无权访问，可能需要刷新 Token"
                            ));
                            Some("FORBIDDEN".to_string())
                        }
                        400 => {
                            warnings.push(format!("{test_name} 返回 400: 请求格式可能已变更"));
                            Some("BAD_REQUEST".to_string())
                        }
                        404 => {
                            warnings.push(format!("{test_name} 返回 404: 模型或接口可能已下线"));
                            Some("NOT_FOUND".to_string())
                        }
                        429 => {
                            warnings.push(format!("{test_name} 返回 429: 请求过于频繁"));
                            Some("RATE_LIMITED".to_string())
                        }
                        500..=599 => {
                            warnings.push(format!("{test_name} 返回 {status}: 服务端错误"));
                            Some("SERVER_ERROR".to_string())
                        }
                        _ => Some("UNKNOWN_ERROR".to_string()),
                    };
                    (
                        false,
                        err_type,
                        Some(body[..body.len().min(200)].to_string()),
                    )
                };

                results.push(ApiCheckResult {
                    model: test_name,
                    available,
                    status,
                    error_type,
                    error_message,
                    time_ms,
                });
            }
            Err(e) => {
                warnings.push(format!("{test_name} 请求失败: {e}"));
                results.push(ApiCheckResult {
                    model: test_name,
                    available: false,
                    status: 0,
                    error_type: Some("REQUEST_FAILED".to_string()),
                    error_message: Some(e.to_string()),
                    time_ms,
                });
            }
        }
    }

    let overall_status = if results.iter().all(|r| r.available) {
        "healthy".to_string()
    } else if results.iter().any(|r| r.available) {
        "partial".to_string()
    } else {
        "error".to_string()
    };

    let checked_at = chrono::Utc::now().to_rfc3339();

    logs.write().await.add(
        "info",
        &format!("[API检测] {provider} 检测完成: {overall_status}"),
    );

    Ok(ApiCompatibilityResult {
        provider,
        overall_status,
        checked_at,
        results,
        warnings,
    })
}

#[tauri::command]
async fn get_available_models() -> Result<Vec<ModelInfo>, String> {
    Ok(vec![
        // Kiro/Claude models
        ModelInfo {
            id: "claude-sonnet-4-5".to_string(),
            object: "model".to_string(),
            owned_by: "anthropic".to_string(),
        },
        ModelInfo {
            id: "claude-sonnet-4-5-20250514".to_string(),
            object: "model".to_string(),
            owned_by: "anthropic".to_string(),
        },
        ModelInfo {
            id: "claude-sonnet-4-5-20250929".to_string(),
            object: "model".to_string(),
            owned_by: "anthropic".to_string(),
        },
        ModelInfo {
            id: "claude-3-7-sonnet-20250219".to_string(),
            object: "model".to_string(),
            owned_by: "anthropic".to_string(),
        },
        ModelInfo {
            id: "claude-3-5-sonnet-latest".to_string(),
            object: "model".to_string(),
            owned_by: "anthropic".to_string(),
        },
        ModelInfo {
            id: "claude-opus-4-5-20250514".to_string(),
            object: "model".to_string(),
            owned_by: "anthropic".to_string(),
        },
        ModelInfo {
            id: "claude-haiku-4-5-20250514".to_string(),
            object: "model".to_string(),
            owned_by: "anthropic".to_string(),
        },
        // Gemini models
        ModelInfo {
            id: "gemini-2.5-flash".to_string(),
            object: "model".to_string(),
            owned_by: "google".to_string(),
        },
        ModelInfo {
            id: "gemini-2.5-flash-lite".to_string(),
            object: "model".to_string(),
            owned_by: "google".to_string(),
        },
        ModelInfo {
            id: "gemini-2.5-pro".to_string(),
            object: "model".to_string(),
            owned_by: "google".to_string(),
        },
        ModelInfo {
            id: "gemini-2.5-pro-preview-06-05".to_string(),
            object: "model".to_string(),
            owned_by: "google".to_string(),
        },
        ModelInfo {
            id: "gemini-3-pro-preview".to_string(),
            object: "model".to_string(),
            owned_by: "google".to_string(),
        },
        // Qwen models
        ModelInfo {
            id: "qwen3-coder-plus".to_string(),
            object: "model".to_string(),
            owned_by: "alibaba".to_string(),
        },
        ModelInfo {
            id: "qwen3-coder-flash".to_string(),
            object: "model".to_string(),
            owned_by: "alibaba".to_string(),
        },
    ])
}

#[tauri::command]
async fn test_api(
    state: tauri::State<'_, AppState>,
    method: String,
    path: String,
    body: Option<String>,
    auth: bool,
) -> Result<TestResult, String> {
    let s = state.read().await;
    let base_url = format!("http://{}:{}", s.config.server.host, s.config.server.port);
    // 优先使用服务器运行时的 API key，确保测试使用的 key 和服务器一致
    // 如果服务器未运行，则使用配置中的 key
    let api_key = s
        .running_api_key
        .as_ref()
        .unwrap_or(&s.config.server.api_key);

    // 创建一个禁用代理的客户端
    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .map_err(|e| e.to_string())?;

    let url = format!("{base_url}{path}");

    tracing::info!("Testing API: {} {}", method, url);

    let start = std::time::Instant::now();

    let mut req = match method.as_str() {
        "GET" => client.get(&url),
        "POST" => client.post(&url),
        _ => return Err("Unsupported method".to_string()),
    };

    req = req.header("Content-Type", "application/json");

    if auth {
        req = req.header("Authorization", format!("Bearer {api_key}"));
    }

    if let Some(b) = body {
        req = req.body(b);
    }

    match req.send().await {
        Ok(resp) => {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            let time_ms = start.elapsed().as_millis() as u64;

            tracing::info!(
                "API test result: status={}, body_len={}",
                status,
                body.len()
            );

            Ok(TestResult {
                success: (200..300).contains(&status),
                status,
                body,
                time_ms,
            })
        }
        Err(e) => {
            tracing::error!("API test error: {}", e);
            Err(e.to_string())
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let config = config::load_config().unwrap_or_default();
    let state: AppState = Arc::new(RwLock::new(server::ServerState::new(config)));
    let logs: LogState = Arc::new(RwLock::new(logger::LogStore::new()));

    // Initialize database for Switch functionality
    let db = database::init_database().expect("Failed to initialize database");

    // Initialize SkillService
    let skill_service = SkillService::new().expect("Failed to initialize SkillService");
    let skill_service_state = SkillServiceState(Arc::new(skill_service));

    // Initialize ProviderPoolService
    let provider_pool_service = ProviderPoolService::new();
    let provider_pool_service_state = ProviderPoolServiceState(Arc::new(provider_pool_service));

    // Initialize CredentialSyncService (optional - only if config manager is available)
    // For now, we initialize it as None since ConfigManager requires async setup
    // This can be enhanced later to properly initialize with ConfigManager
    let credential_sync_service_state = CredentialSyncServiceState(None);

    // Initialize TokenCacheService
    let token_cache_service = TokenCacheService::new();
    let token_cache_service_state = TokenCacheServiceState(Arc::new(token_cache_service));

    // Initialize RouterConfigState
    let router_config_state = RouterConfigState::default();

    // Initialize ResilienceConfigState
    let resilience_config_state = ResilienceConfigState::default();

    // Initialize shared telemetry instances for both TelemetryState and RequestProcessor
    // This allows the frontend monitoring page to display data recorded by the request processor
    let shared_stats = Arc::new(parking_lot::RwLock::new(
        telemetry::StatsAggregator::with_defaults(),
    ));
    let shared_tokens = Arc::new(parking_lot::RwLock::new(
        telemetry::TokenTracker::with_defaults(),
    ));
    let shared_logger = Arc::new(
        telemetry::RequestLogger::with_defaults().expect("Failed to create RequestLogger"),
    );

    // Initialize TelemetryState with shared instances
    let telemetry_state = commands::telemetry_cmd::TelemetryState::with_shared(
        shared_stats.clone(),
        shared_tokens.clone(),
        Some(shared_logger.clone()),
    )
    .expect("Failed to create TelemetryState");

    // Initialize default skill repos
    {
        let conn = db.lock().expect("Failed to lock database");
        database::dao::skills::SkillDao::init_default_skill_repos(&conn)
            .expect("Failed to initialize default skill repos");
    }

    // Clone for setup hook
    let state_clone = state.clone();
    let logs_clone = logs.clone();
    let db_clone = db.clone();
    let pool_service_clone = provider_pool_service_state.0.clone();
    let token_cache_clone = token_cache_service_state.0.clone();
    let shared_stats_clone = shared_stats.clone();
    let shared_tokens_clone = shared_tokens.clone();
    let shared_logger_clone = shared_logger.clone();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            Some(vec!["--minimized"]),
        ))
        .manage(state)
        .manage(logs)
        .manage(db)
        .manage(skill_service_state)
        .manage(provider_pool_service_state)
        .manage(credential_sync_service_state)
        .manage(token_cache_service_state)
        .manage(router_config_state)
        .manage(resilience_config_state)
        .manage(telemetry_state)
        .setup(move |app| {
            // 初始化托盘管理器
            // Requirements 1.4: 应用启动时显示停止状态图标
            match TrayManager::new(app.handle()) {
                Ok(tray_manager) => {
                    tracing::info!("[启动] 托盘管理器初始化成功");
                    // 将托盘管理器存储到应用状态中
                    let tray_state: TrayManagerState<tauri::Wry> =
                        TrayManagerState(Arc::new(tokio::sync::RwLock::new(Some(tray_manager))));
                    app.manage(tray_state);
                }
                Err(e) => {
                    tracing::error!("[启动] 托盘管理器初始化失败: {}", e);
                    // 即使托盘初始化失败，应用仍然可以运行
                    let tray_state: TrayManagerState<tauri::Wry> =
                        TrayManagerState(Arc::new(tokio::sync::RwLock::new(None)));
                    app.manage(tray_state);
                }
            }
            // 自动启动服务器
            let state = state_clone.clone();
            let logs = logs_clone.clone();
            let db = db_clone.clone();
            let pool_service = pool_service_clone.clone();
            let token_cache = token_cache_clone.clone();
            let shared_stats = shared_stats_clone.clone();
            let shared_tokens = shared_tokens_clone.clone();
            let shared_logger = shared_logger_clone.clone();
            let app_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                // 先加载凭证
                {
                    let mut s = state.write().await;
                    if let Err(e) = s.kiro_provider.load_credentials().await {
                        logs.write()
                            .await
                            .add("warn", &format!("[启动] 加载 Kiro 凭证失败: {e}"));
                    } else {
                        logs.write().await.add("info", "[启动] Kiro 凭证已加载");
                    }
                }
                // 启动服务器（使用共享的遥测实例）
                let server_started;
                let server_address;
                {
                    let mut s = state.write().await;
                    logs.write()
                        .await
                        .add("info", "[启动] 正在自动启动服务器...");
                    match s
                        .start_with_telemetry(
                            logs.clone(),
                            pool_service,
                            token_cache,
                            Some(db),
                            Some(shared_stats),
                            Some(shared_tokens),
                            Some(shared_logger),
                        )
                        .await
                    {
                        Ok(_) => {
                            let host = s.config.server.host.clone();
                            let port = s.config.server.port;
                            logs.write()
                                .await
                                .add("info", &format!("[启动] 服务器已启动: {host}:{port}"));
                            server_started = true;
                            server_address = format!("{}:{}", host, port);
                        }
                        Err(e) => {
                            logs.write()
                                .await
                                .add("error", &format!("[启动] 服务器启动失败: {e}"));
                            server_started = false;
                            server_address = String::new();
                        }
                    }
                }

                // 更新托盘状态
                // Requirements 7.1: API 服务器状态变化时更新托盘图标
                if let Some(tray_state) = app_handle.try_state::<TrayManagerState<tauri::Wry>>() {
                    let tray_guard = tray_state.0.read().await;
                    if let Some(tray_manager) = tray_guard.as_ref() {
                        // 计算初始图标状态
                        // 服务器刚启动时，假设凭证健康（后续会通过状态同步更新）
                        let icon_status = if server_started {
                            TrayIconStatus::Running
                        } else {
                            TrayIconStatus::Stopped
                        };

                        let snapshot = TrayStateSnapshot {
                            icon_status,
                            server_running: server_started,
                            server_address,
                            available_credentials: 0, // 初始值，后续通过状态同步更新
                            total_credentials: 0,
                            today_requests: 0,
                            auto_start_enabled: false, // 后续通过状态同步更新
                        };

                        if let Err(e) = tray_manager.update_state(snapshot).await {
                            tracing::error!("[启动] 更新托盘状态失败: {}", e);
                        } else {
                            tracing::info!("[启动] 托盘状态已更新");
                        }
                    }
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            start_server,
            stop_server,
            get_server_status,
            get_config,
            save_config,
            get_default_provider,
            set_default_provider,
            // Unified OAuth commands (new)
            commands::oauth_cmd::get_oauth_credentials,
            commands::oauth_cmd::reload_oauth_credentials,
            commands::oauth_cmd::refresh_oauth_token,
            commands::oauth_cmd::get_oauth_env_variables,
            commands::oauth_cmd::get_oauth_token_file_hash,
            commands::oauth_cmd::check_and_reload_oauth_credentials,
            commands::oauth_cmd::get_all_oauth_credentials,
            // Legacy Kiro commands (deprecated, kept for compatibility)
            refresh_kiro_token,
            reload_credentials,
            get_kiro_credentials,
            get_env_variables,
            get_token_file_hash,
            check_and_reload_credentials,
            // Legacy Gemini commands (deprecated, kept for compatibility)
            get_gemini_credentials,
            reload_gemini_credentials,
            refresh_gemini_token,
            get_gemini_env_variables,
            get_gemini_token_file_hash,
            check_and_reload_gemini_credentials,
            // Legacy Qwen commands (deprecated, kept for compatibility)
            get_qwen_credentials,
            reload_qwen_credentials,
            refresh_qwen_token,
            get_qwen_env_variables,
            get_qwen_token_file_hash,
            check_and_reload_qwen_credentials,
            // OpenAI Custom commands
            get_openai_custom_status,
            set_openai_custom_config,
            // Claude Custom commands
            get_claude_custom_status,
            set_claude_custom_config,
            // Common
            get_logs,
            clear_logs,
            test_api,
            get_available_models,
            // API Compatibility
            check_api_compatibility,
            // Switch commands
            commands::switch_cmd::get_switch_providers,
            commands::switch_cmd::get_current_switch_provider,
            commands::switch_cmd::add_switch_provider,
            commands::switch_cmd::update_switch_provider,
            commands::switch_cmd::delete_switch_provider,
            commands::switch_cmd::switch_provider,
            commands::switch_cmd::import_default_config,
            commands::switch_cmd::read_live_provider_settings,
            // Config commands
            commands::config_cmd::get_config_status,
            commands::config_cmd::get_config_dir_path,
            commands::config_cmd::open_config_folder,
            commands::config_cmd::get_tool_versions,
            commands::config_cmd::get_auto_launch_status,
            commands::config_cmd::set_auto_launch,
            // Config import/export commands
            commands::config_cmd::export_config,
            commands::config_cmd::validate_config_yaml,
            commands::config_cmd::import_config,
            commands::config_cmd::get_config_paths,
            // Enhanced export/import commands (using ExportService/ImportService)
            commands::config_cmd::export_bundle,
            commands::config_cmd::export_config_yaml,
            commands::config_cmd::validate_import,
            commands::config_cmd::import_bundle,
            // Path utility commands
            commands::config_cmd::expand_path,
            commands::config_cmd::open_auth_dir,
            // Update check command
            commands::config_cmd::check_for_updates,
            // MCP commands
            commands::mcp_cmd::get_mcp_servers,
            commands::mcp_cmd::add_mcp_server,
            commands::mcp_cmd::update_mcp_server,
            commands::mcp_cmd::delete_mcp_server,
            commands::mcp_cmd::toggle_mcp_server,
            commands::mcp_cmd::import_mcp_from_app,
            commands::mcp_cmd::sync_all_mcp_to_live,
            // Prompt commands
            commands::prompt_cmd::get_prompts,
            commands::prompt_cmd::upsert_prompt,
            commands::prompt_cmd::add_prompt,
            commands::prompt_cmd::update_prompt,
            commands::prompt_cmd::delete_prompt,
            commands::prompt_cmd::enable_prompt,
            commands::prompt_cmd::import_prompt_from_file,
            commands::prompt_cmd::get_current_prompt_file_content,
            commands::prompt_cmd::auto_import_prompt,
            commands::prompt_cmd::switch_prompt,
            // Skill commands
            commands::skill_cmd::get_skills,
            commands::skill_cmd::get_skills_for_app,
            commands::skill_cmd::install_skill,
            commands::skill_cmd::install_skill_for_app,
            commands::skill_cmd::uninstall_skill,
            commands::skill_cmd::uninstall_skill_for_app,
            commands::skill_cmd::get_skill_repos,
            commands::skill_cmd::add_skill_repo,
            commands::skill_cmd::remove_skill_repo,
            // Provider Pool commands
            commands::provider_pool_cmd::get_provider_pool_overview,
            commands::provider_pool_cmd::get_provider_pool_credentials,
            commands::provider_pool_cmd::add_provider_pool_credential,
            commands::provider_pool_cmd::update_provider_pool_credential,
            commands::provider_pool_cmd::delete_provider_pool_credential,
            commands::provider_pool_cmd::toggle_provider_pool_credential,
            commands::provider_pool_cmd::reset_provider_pool_credential,
            commands::provider_pool_cmd::reset_provider_pool_health,
            commands::provider_pool_cmd::check_provider_pool_credential_health,
            commands::provider_pool_cmd::check_provider_pool_type_health,
            commands::provider_pool_cmd::add_kiro_oauth_credential,
            commands::provider_pool_cmd::add_gemini_oauth_credential,
            commands::provider_pool_cmd::add_qwen_oauth_credential,
            commands::provider_pool_cmd::add_antigravity_oauth_credential,
            commands::provider_pool_cmd::add_openai_key_credential,
            commands::provider_pool_cmd::add_claude_key_credential,
            commands::provider_pool_cmd::refresh_pool_credential_token,
            commands::provider_pool_cmd::get_pool_credential_oauth_status,
            commands::provider_pool_cmd::debug_kiro_credentials,
            commands::provider_pool_cmd::test_user_credentials,
            commands::provider_pool_cmd::migrate_private_config_to_pool,
            // Route commands
            commands::route_cmd::get_available_routes,
            commands::route_cmd::get_route_curl_examples,
            // Router config commands
            commands::router_cmd::get_model_aliases,
            commands::router_cmd::add_model_alias,
            commands::router_cmd::remove_model_alias,
            commands::router_cmd::get_routing_rules,
            commands::router_cmd::add_routing_rule,
            commands::router_cmd::remove_routing_rule,
            commands::router_cmd::update_routing_rule,
            commands::router_cmd::get_exclusions,
            commands::router_cmd::add_exclusion,
            commands::router_cmd::remove_exclusion,
            commands::router_cmd::set_router_default_provider,
            commands::router_cmd::get_recommended_presets,
            commands::router_cmd::apply_recommended_preset,
            commands::router_cmd::clear_all_routing_config,
            // Resilience config commands
            commands::resilience_cmd::get_retry_config,
            commands::resilience_cmd::update_retry_config,
            commands::resilience_cmd::get_failover_config,
            commands::resilience_cmd::update_failover_config,
            commands::resilience_cmd::get_switch_log,
            commands::resilience_cmd::clear_switch_log,
            // Telemetry commands
            commands::telemetry_cmd::get_request_logs,
            commands::telemetry_cmd::get_request_log_detail,
            commands::telemetry_cmd::clear_request_logs,
            commands::telemetry_cmd::get_stats_summary,
            commands::telemetry_cmd::get_stats_by_provider,
            commands::telemetry_cmd::get_stats_by_model,
            commands::telemetry_cmd::get_token_summary,
            commands::telemetry_cmd::get_token_stats_by_provider,
            commands::telemetry_cmd::get_token_stats_by_model,
            commands::telemetry_cmd::get_token_stats_by_day,
            commands::telemetry_cmd::get_dashboard_data,
            // Injection commands
            commands::injection_cmd::get_injection_config,
            commands::injection_cmd::set_injection_enabled,
            commands::injection_cmd::get_injection_rules,
            commands::injection_cmd::add_injection_rule,
            commands::injection_cmd::remove_injection_rule,
            commands::injection_cmd::update_injection_rule,
            // Usage commands
            commands::usage_cmd::get_kiro_usage,
            // Tray commands
            commands::tray_cmd::sync_tray_state,
            commands::tray_cmd::update_tray_server_status,
            commands::tray_cmd::update_tray_credential_status,
            commands::tray_cmd::get_tray_state,
            commands::tray_cmd::refresh_tray_menu,
            commands::tray_cmd::refresh_tray_with_stats,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
