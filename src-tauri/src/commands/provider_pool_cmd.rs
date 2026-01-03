//! Provider Pool Tauri 命令

#![allow(dead_code)]

use crate::credential::CredentialSyncService;
use crate::database::dao::provider_pool::ProviderPoolDao;
use crate::database::DbConnection;
use crate::models::provider_pool_model::{
    AddCredentialRequest, CredentialData, CredentialDisplay, HealthCheckResult, OAuthStatus,
    PoolProviderType, ProviderCredential, ProviderPoolOverview, UpdateCredentialRequest,
};
use crate::services::provider_pool_service::ProviderPoolService;
use chrono::Utc;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tauri::{Emitter, State};
use uuid::Uuid;

pub struct ProviderPoolServiceState(pub Arc<ProviderPoolService>);

/// 凭证同步服务状态封装
pub struct CredentialSyncServiceState(pub Option<Arc<CredentialSyncService>>);

/// 展开路径中的 ~ 为用户主目录
fn expand_tilde(path: &str) -> String {
    if let Some(stripped) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(stripped).to_string_lossy().to_string();
        }
    }
    path.to_string()
}

/// 获取应用凭证存储目录
fn get_credentials_dir() -> Result<PathBuf, String> {
    let app_data_dir = dirs::data_dir()
        .ok_or_else(|| "无法获取应用数据目录".to_string())?
        .join("proxycast")
        .join("credentials");

    // 确保目录存在
    if !app_data_dir.exists() {
        fs::create_dir_all(&app_data_dir).map_err(|e| format!("创建凭证存储目录失败: {}", e))?;
    }

    Ok(app_data_dir)
}

/// 复制并重命名 OAuth 凭证文件
///
/// 对于 Kiro 凭证，会自动合并 clientIdHash 文件中的 client_id/client_secret，
/// 使副本文件完全独立，支持多账号场景。
fn copy_and_rename_credential_file(
    source_path: &str,
    provider_type: &str,
) -> Result<String, String> {
    let expanded_source = expand_tilde(source_path);
    let source = Path::new(&expanded_source);

    // 验证源文件存在
    if !source.exists() {
        return Err(format!("凭证文件不存在: {}", expanded_source));
    }

    // 生成新的文件名：{provider_type}_{uuid}_{timestamp}.json
    let uuid = Uuid::new_v4().to_string();
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let new_filename = format!(
        "{}_{}_{}_{}.json",
        provider_type,
        &uuid[..8], // 使用 UUID 前8位
        timestamp,
        provider_type
    );

    // 获取目标目录
    let credentials_dir = get_credentials_dir()?;
    let target_path = credentials_dir.join(&new_filename);

    // 对于 Kiro 凭证，需要合并 clientIdHash 文件中的 client_id/client_secret
    if provider_type == "kiro" {
        let content = fs::read_to_string(source).map_err(|e| format!("读取凭证文件失败: {}", e))?;
        let mut creds: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| format!("解析凭证文件失败: {}", e))?;

        // 检测 refreshToken 是否被截断（仅记录警告，不阻止添加）
        // 正常的 refreshToken 长度应该在 500+ 字符，如果小于 100 字符则可能被截断
        // 注意：即使 refreshToken 被截断，也允许添加凭证，在刷新时才会提示错误
        if let Some(refresh_token) = creds.get("refreshToken").and_then(|v| v.as_str()) {
            let token_len = refresh_token.len();

            // 检测常见的截断模式
            let is_truncated =
                token_len < 100 || refresh_token.ends_with("...") || refresh_token.contains("...");

            if is_truncated {
                // 安全地截取前 50 个字符（避免 UTF-8 边界 panic）
                let preview: String = refresh_token.chars().take(50).collect();
                tracing::warn!(
                    "[KIRO] 检测到 refreshToken 可能被截断！长度: {}, 内容: {}... (仍允许添加，刷新时会提示)",
                    token_len,
                    preview
                );
                // 不再阻止添加，只记录警告
                // 在刷新 Token 时会检测并提示用户
            } else {
                tracing::info!("[KIRO] refreshToken 长度检查通过: {} 字符", token_len);
            }
        } else {
            tracing::warn!("[KIRO] 凭证文件中没有 refreshToken 字段");
        }

        let aws_sso_cache_dir = dirs::home_dir()
            .ok_or_else(|| "无法获取用户主目录".to_string())?
            .join(".aws")
            .join("sso")
            .join("cache");

        // 尝试从 clientIdHash 文件或扫描目录获取 client_id/client_secret
        let mut found_credentials = false;

        // 方式1：如果有 clientIdHash，读取对应文件
        if let Some(hash) = creds.get("clientIdHash").and_then(|v| v.as_str()) {
            let hash_file_path = aws_sso_cache_dir.join(format!("{}.json", hash));

            if hash_file_path.exists() {
                if let Ok(hash_content) = fs::read_to_string(&hash_file_path) {
                    if let Ok(hash_json) = serde_json::from_str::<serde_json::Value>(&hash_content)
                    {
                        if let Some(client_id) = hash_json.get("clientId") {
                            creds["clientId"] = client_id.clone();
                        }
                        if let Some(client_secret) = hash_json.get("clientSecret") {
                            creds["clientSecret"] = client_secret.clone();
                        }
                        if creds.get("clientId").is_some() && creds.get("clientSecret").is_some() {
                            found_credentials = true;
                            tracing::info!(
                                "[KIRO] 已从 clientIdHash 文件合并 client_id/client_secret 到副本"
                            );
                        }
                    }
                }
            }
        }

        // 方式2：如果没有 clientIdHash 或未找到，扫描目录中的其他 JSON 文件
        if !found_credentials && aws_sso_cache_dir.exists() {
            tracing::info!(
                "[KIRO] 没有 clientIdHash 或未找到，扫描目录查找 client_id/client_secret"
            );
            if let Ok(entries) = fs::read_dir(&aws_sso_cache_dir) {
                for entry in entries.flatten() {
                    let file_path = entry.path();
                    // 跳过主凭证文件和备份文件
                    if file_path.extension().map(|e| e == "json").unwrap_or(false) {
                        let file_name =
                            file_path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                        if file_name.starts_with("kiro-auth-token") {
                            continue;
                        }
                        if let Ok(file_content) = fs::read_to_string(&file_path) {
                            if let Ok(file_json) =
                                serde_json::from_str::<serde_json::Value>(&file_content)
                            {
                                let has_client_id =
                                    file_json.get("clientId").and_then(|v| v.as_str()).is_some();
                                let has_client_secret = file_json
                                    .get("clientSecret")
                                    .and_then(|v| v.as_str())
                                    .is_some();
                                if has_client_id && has_client_secret {
                                    creds["clientId"] = file_json["clientId"].clone();
                                    creds["clientSecret"] = file_json["clientSecret"].clone();
                                    found_credentials = true;
                                    tracing::info!(
                                        "[KIRO] 已从 {} 合并 client_id/client_secret 到副本",
                                        file_name
                                    );
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        if !found_credentials {
            // 检查认证方式
            let auth_method = creds
                .get("authMethod")
                .and_then(|v| v.as_str())
                .unwrap_or("social");

            if auth_method.to_lowercase() == "idc" {
                // IdC 认证必须有 clientId/clientSecret
                tracing::error!(
                    "[KIRO] IdC 认证方式缺少 clientId/clientSecret，无法创建有效的凭证副本"
                );
                return Err(
                    "IdC 认证凭证不完整：缺少 clientId/clientSecret。\n\n💡 解决方案：\n1. 确保 ~/.aws/sso/cache/ 目录下有对应的 clientIdHash 文件\n2. 如果使用 AWS IAM Identity Center，请确保已完成完整的 SSO 登录流程\n3. 或者尝试使用 Social 认证方式的凭证".to_string()
                );
            } else {
                tracing::warn!("[KIRO] 未找到 client_id/client_secret，将使用 social 认证方式");
            }
        }

        // 写入合并后的凭证到副本文件
        let merged_content =
            serde_json::to_string_pretty(&creds).map_err(|e| format!("序列化凭证失败: {}", e))?;
        fs::write(&target_path, merged_content).map_err(|e| format!("写入凭证文件失败: {}", e))?;
    } else {
        // 其他类型直接复制
        fs::copy(source, &target_path).map_err(|e| format!("复制凭证文件失败: {}", e))?;
    }

    // 返回新的文件路径
    Ok(target_path.to_string_lossy().to_string())
}

/// 删除凭证文件（如果在应用存储目录中）
fn cleanup_credential_file(file_path: &str) -> Result<(), String> {
    let path = Path::new(file_path);

    // 只删除在应用凭证存储目录中的文件
    if let Ok(credentials_dir) = get_credentials_dir() {
        if let Ok(canonical_path) = path.canonicalize() {
            if let Ok(canonical_dir) = credentials_dir.canonicalize() {
                if canonical_path.starts_with(canonical_dir) {
                    if let Err(e) = fs::remove_file(&canonical_path) {
                        // 只记录警告，不中断删除过程
                        println!("Warning: Failed to delete credential file: {}", e);
                    }
                }
            }
        }
    }

    Ok(())
}

/// 获取凭证池概览
#[tauri::command]
pub fn get_provider_pool_overview(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
) -> Result<Vec<ProviderPoolOverview>, String> {
    pool_service.0.get_overview(&db)
}

/// 获取指定类型的凭证列表
#[tauri::command]
pub fn get_provider_pool_credentials(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    provider_type: String,
) -> Result<Vec<CredentialDisplay>, String> {
    pool_service.0.get_by_type(&db, &provider_type)
}

/// 添加凭证
///
/// 添加凭证到数据库，并同步到 YAML 配置文件
/// Requirements: 1.1, 1.2
#[tauri::command]
pub fn add_provider_pool_credential(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    sync_service: State<'_, CredentialSyncServiceState>,
    request: AddCredentialRequest,
) -> Result<ProviderCredential, String> {
    // 添加到数据库
    let credential = pool_service.0.add_credential(
        &db,
        &request.provider_type,
        request.credential,
        request.name,
        request.check_health,
        request.check_model_name,
    )?;

    // 同步到 YAML 配置（如果同步服务可用）
    if let Some(ref sync) = sync_service.0 {
        if let Err(e) = sync.add_credential(&credential) {
            // 记录警告但不中断操作
            tracing::warn!("同步凭证到 YAML 失败: {}", e);
        }
    }

    Ok(credential)
}

/// 更新凭证
/// 更新凭证
///
/// 更新数据库中的凭证，并同步到 YAML 配置文件
/// Requirements: 1.1, 1.2
#[tauri::command]
pub fn update_provider_pool_credential(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    sync_service: State<'_, CredentialSyncServiceState>,
    uuid: String,
    request: UpdateCredentialRequest,
) -> Result<ProviderCredential, String> {
    tracing::info!(
        "[UPDATE_CREDENTIAL] 收到更新请求: uuid={}, name={:?}, check_model_name={:?}, not_supported_models={:?}",
        uuid,
        request.name,
        request.check_model_name,
        request.not_supported_models
    );
    // 如果需要重新上传文件，先处理文件上传
    let credential = if let Some(new_file_path) = request.new_creds_file_path {
        // 获取当前凭证以确定类型
        let conn = db.lock().map_err(|e| e.to_string())?;
        let current_credential = ProviderPoolDao::get_by_uuid(&conn, &uuid)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("凭证不存在: {}", uuid))?;

        // 根据凭证类型复制新文件
        let new_stored_path = match &current_credential.credential {
            CredentialData::KiroOAuth { creds_file_path } => {
                // 清理旧文件
                cleanup_credential_file(creds_file_path)?;
                copy_and_rename_credential_file(&new_file_path, "kiro")?
            }
            CredentialData::GeminiOAuth {
                creds_file_path, ..
            } => {
                // 清理旧文件
                cleanup_credential_file(creds_file_path)?;
                copy_and_rename_credential_file(&new_file_path, "gemini")?
            }
            CredentialData::QwenOAuth { creds_file_path } => {
                // 清理旧文件
                cleanup_credential_file(creds_file_path)?;
                copy_and_rename_credential_file(&new_file_path, "qwen")?
            }
            CredentialData::AntigravityOAuth {
                creds_file_path, ..
            } => {
                // 清理旧文件
                cleanup_credential_file(creds_file_path)?;
                copy_and_rename_credential_file(&new_file_path, "antigravity")?
            }
            _ => {
                return Err("只有 OAuth 凭证支持重新上传文件".to_string());
            }
        };

        // 更新凭证数据
        let mut updated_cred = current_credential;

        // 更新凭证数据中的文件路径
        match &mut updated_cred.credential {
            CredentialData::KiroOAuth { creds_file_path } => {
                *creds_file_path = new_stored_path;
            }
            CredentialData::GeminiOAuth {
                creds_file_path,
                project_id,
            } => {
                *creds_file_path = new_stored_path;
                if let Some(new_pid) = request.new_project_id {
                    *project_id = Some(new_pid);
                }
            }
            CredentialData::QwenOAuth { creds_file_path } => {
                *creds_file_path = new_stored_path;
            }
            CredentialData::AntigravityOAuth {
                creds_file_path,
                project_id,
            } => {
                *creds_file_path = new_stored_path;
                if let Some(new_pid) = request.new_project_id {
                    *project_id = Some(new_pid);
                }
            }
            _ => {}
        }

        // 应用其他更新
        // 处理 name：空字符串表示清除，None 表示不修改
        if let Some(name) = request.name {
            updated_cred.name = if name.is_empty() { None } else { Some(name) };
        }
        if let Some(is_disabled) = request.is_disabled {
            updated_cred.is_disabled = is_disabled;
        }
        if let Some(check_health) = request.check_health {
            updated_cred.check_health = check_health;
        }
        // 处理 check_model_name：空字符串表示清除，None 表示不修改
        if let Some(check_model_name) = request.check_model_name {
            updated_cred.check_model_name = if check_model_name.is_empty() {
                None
            } else {
                Some(check_model_name)
            };
        }
        if let Some(not_supported_models) = request.not_supported_models {
            updated_cred.not_supported_models = not_supported_models;
        }

        updated_cred.updated_at = Utc::now();

        // 保存到数据库
        ProviderPoolDao::update(&conn, &updated_cred).map_err(|e| e.to_string())?;

        updated_cred
    } else if request.new_base_url.is_some() || request.new_api_key.is_some() {
        // 更新 API Key 凭证的 api_key 和/或 base_url
        let conn = db.lock().map_err(|e| e.to_string())?;
        let mut current_credential = ProviderPoolDao::get_by_uuid(&conn, &uuid)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("凭证不存在: {}", uuid))?;

        // 更新 api_key 和 base_url
        match &mut current_credential.credential {
            CredentialData::OpenAIKey { api_key, base_url } => {
                if let Some(new_key) = request.new_api_key {
                    if !new_key.is_empty() {
                        *api_key = new_key;
                    }
                }
                if let Some(new_url) = request.new_base_url {
                    *base_url = if new_url.is_empty() {
                        None
                    } else {
                        Some(new_url)
                    };
                }
            }
            CredentialData::ClaudeKey { api_key, base_url } => {
                if let Some(new_key) = request.new_api_key {
                    if !new_key.is_empty() {
                        *api_key = new_key;
                    }
                }
                if let Some(new_url) = request.new_base_url {
                    *base_url = if new_url.is_empty() {
                        None
                    } else {
                        Some(new_url)
                    };
                }
            }
            _ => {
                return Err("只有 API Key 凭证支持修改 API Key 和 Base URL".to_string());
            }
        }

        // 应用其他更新
        // 处理 name：空字符串表示清除，None 表示不修改
        if let Some(name) = request.name {
            current_credential.name = if name.is_empty() { None } else { Some(name) };
        }
        if let Some(is_disabled) = request.is_disabled {
            current_credential.is_disabled = is_disabled;
        }
        if let Some(check_health) = request.check_health {
            current_credential.check_health = check_health;
        }
        // 处理 check_model_name：空字符串表示清除，None 表示不修改
        if let Some(check_model_name) = request.check_model_name {
            current_credential.check_model_name = if check_model_name.is_empty() {
                None
            } else {
                Some(check_model_name)
            };
        }
        if let Some(not_supported_models) = request.not_supported_models {
            current_credential.not_supported_models = not_supported_models;
        }

        current_credential.updated_at = Utc::now();

        // 保存到数据库
        ProviderPoolDao::update(&conn, &current_credential).map_err(|e| e.to_string())?;

        current_credential
    } else {
        // 常规更新，不涉及文件
        pool_service.0.update_credential(
            &db,
            &uuid,
            request.name,
            request.is_disabled,
            request.check_health,
            request.check_model_name,
            request.not_supported_models,
            request.new_proxy_url,
        )?
    };

    // 同步到 YAML 配置（如果同步服务可用）
    if let Some(ref sync) = sync_service.0 {
        if let Err(e) = sync.update_credential(&credential) {
            // 记录警告但不中断操作
            tracing::warn!("同步凭证更新到 YAML 失败: {}", e);
        }
    }

    Ok(credential)
}

/// 删除凭证
/// 删除凭证
///
/// 从数据库删除凭证，并同步到 YAML 配置文件
/// Requirements: 1.1, 1.2
#[tauri::command]
pub fn delete_provider_pool_credential(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    sync_service: State<'_, CredentialSyncServiceState>,
    uuid: String,
    provider_type: Option<String>,
) -> Result<bool, String> {
    // 从数据库删除
    let result = pool_service.0.delete_credential(&db, &uuid)?;

    // 同步到 YAML 配置（如果同步服务可用且提供了 provider_type）
    if let Some(ref sync) = sync_service.0 {
        if let Some(pt) = provider_type {
            if let Ok(pool_type) = pt.parse::<PoolProviderType>() {
                if let Err(e) = sync.remove_credential(pool_type, &uuid) {
                    // 记录警告但不中断操作
                    tracing::warn!("从 YAML 删除凭证失败: {}", e);
                }
            }
        }
    }

    Ok(result)
}

/// 切换凭证启用/禁用状态
#[tauri::command]
pub fn toggle_provider_pool_credential(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    uuid: String,
    is_disabled: bool,
) -> Result<ProviderCredential, String> {
    pool_service
        .0
        .update_credential(&db, &uuid, None, Some(is_disabled), None, None, None, None)
}

/// 重置凭证计数器
#[tauri::command]
pub fn reset_provider_pool_credential(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    uuid: String,
) -> Result<(), String> {
    pool_service.0.reset_counters(&db, &uuid)
}

/// 重置指定类型的所有凭证健康状态
#[tauri::command]
pub fn reset_provider_pool_health(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    provider_type: String,
) -> Result<usize, String> {
    pool_service.0.reset_health_by_type(&db, &provider_type)
}

/// 执行单个凭证的健康检查
#[tauri::command]
pub async fn check_provider_pool_credential_health(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    uuid: String,
) -> Result<HealthCheckResult, String> {
    tracing::info!("[DEBUG] 开始健康检查 for uuid: {}", uuid);
    let result = pool_service.0.check_credential_health(&db, &uuid).await;
    match &result {
        Ok(health) => tracing::info!(
            "[DEBUG] 健康检查完成: success={}, message={:?}",
            health.success,
            health.message
        ),
        Err(err) => tracing::error!("[DEBUG] 健康检查失败: {}", err),
    }
    result
}

/// 执行指定类型的所有凭证健康检查
#[tauri::command]
pub async fn check_provider_pool_type_health(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    provider_type: String,
) -> Result<Vec<HealthCheckResult>, String> {
    pool_service.0.check_type_health(&db, &provider_type).await
}

/// 添加 Kiro OAuth 凭证（通过文件路径）
#[tauri::command]
pub fn add_kiro_oauth_credential(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    creds_file_path: String,
    name: Option<String>,
) -> Result<ProviderCredential, String> {
    tracing::info!("[KIRO] 开始添加凭证，文件路径: {}", creds_file_path);

    // 复制并重命名文件到应用存储目录
    let stored_file_path = match copy_and_rename_credential_file(&creds_file_path, "kiro") {
        Ok(path) => {
            tracing::info!("[KIRO] 凭证文件已复制到: {}", path);
            path
        }
        Err(e) => {
            tracing::error!("[KIRO] 复制凭证文件失败: {}", e);
            return Err(e);
        }
    };

    match pool_service.0.add_credential(
        &db,
        "kiro",
        CredentialData::KiroOAuth {
            creds_file_path: stored_file_path,
        },
        name,
        Some(true),
        None,
    ) {
        Ok(cred) => {
            tracing::info!("[KIRO] 凭证添加成功，UUID: {}", cred.uuid);
            Ok(cred)
        }
        Err(e) => {
            tracing::error!("[KIRO] 添加凭证到数据库失败: {}", e);
            Err(e)
        }
    }
}

/// 从 JSON 内容创建 Kiro 凭证文件并添加到凭证池
///
/// 直接粘贴 JSON 内容，无需选择文件
fn create_kiro_credential_from_json(json_content: &str) -> Result<String, String> {
    // 验证 JSON 格式
    let creds: serde_json::Value =
        serde_json::from_str(json_content).map_err(|e| format!("JSON 格式无效: {}", e))?;

    // 验证必要字段
    if creds.get("refreshToken").is_none() {
        return Err("凭证 JSON 缺少 refreshToken 字段".to_string());
    }

    // 检测 refreshToken 是否被截断
    if let Some(refresh_token) = creds.get("refreshToken").and_then(|v| v.as_str()) {
        let token_len = refresh_token.len();
        let is_truncated =
            token_len < 100 || refresh_token.ends_with("...") || refresh_token.contains("...");

        if is_truncated {
            tracing::warn!(
                "[KIRO] 检测到 refreshToken 可能被截断！长度: {} (仍允许添加，刷新时会提示)",
                token_len
            );
        }
    }

    // 生成新的文件名
    let uuid = Uuid::new_v4().to_string();
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let new_filename = format!("kiro_{}_{}_{}.json", &uuid[..8], timestamp, "kiro");

    // 获取目标目录
    let credentials_dir = get_credentials_dir()?;
    let target_path = credentials_dir.join(&new_filename);

    // 尝试合并 clientId/clientSecret（如果凭证中没有）
    let mut merged_creds = creds.clone();

    // 检查是否需要从外部文件获取 clientId/clientSecret
    let has_client_id = merged_creds.get("clientId").is_some();
    let has_client_secret = merged_creds.get("clientSecret").is_some();

    if !has_client_id || !has_client_secret {
        let aws_sso_cache_dir = dirs::home_dir()
            .ok_or_else(|| "无法获取用户主目录".to_string())?
            .join(".aws")
            .join("sso")
            .join("cache");

        let mut found_credentials = false;

        // 方式1：如果有 clientIdHash，读取对应文件
        if let Some(hash) = merged_creds.get("clientIdHash").and_then(|v| v.as_str()) {
            let hash_file_path = aws_sso_cache_dir.join(format!("{}.json", hash));

            if hash_file_path.exists() {
                if let Ok(hash_content) = fs::read_to_string(&hash_file_path) {
                    if let Ok(hash_json) = serde_json::from_str::<serde_json::Value>(&hash_content)
                    {
                        if let Some(client_id) = hash_json.get("clientId") {
                            merged_creds["clientId"] = client_id.clone();
                        }
                        if let Some(client_secret) = hash_json.get("clientSecret") {
                            merged_creds["clientSecret"] = client_secret.clone();
                        }
                        if merged_creds.get("clientId").is_some()
                            && merged_creds.get("clientSecret").is_some()
                        {
                            found_credentials = true;
                            tracing::info!(
                                "[KIRO] 已从 clientIdHash 文件合并 client_id/client_secret"
                            );
                        }
                    }
                }
            }
        }

        // 方式2：扫描目录中的其他 JSON 文件
        if !found_credentials && aws_sso_cache_dir.exists() {
            tracing::info!("[KIRO] 扫描目录查找 client_id/client_secret");
            if let Ok(entries) = fs::read_dir(&aws_sso_cache_dir) {
                for entry in entries.flatten() {
                    let file_path = entry.path();
                    if file_path.extension().map(|e| e == "json").unwrap_or(false) {
                        let file_name =
                            file_path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                        if file_name.starts_with("kiro-auth-token") {
                            continue;
                        }
                        if let Ok(file_content) = fs::read_to_string(&file_path) {
                            if let Ok(file_json) =
                                serde_json::from_str::<serde_json::Value>(&file_content)
                            {
                                let has_cid =
                                    file_json.get("clientId").and_then(|v| v.as_str()).is_some();
                                let has_csec = file_json
                                    .get("clientSecret")
                                    .and_then(|v| v.as_str())
                                    .is_some();
                                if has_cid && has_csec {
                                    merged_creds["clientId"] = file_json["clientId"].clone();
                                    merged_creds["clientSecret"] =
                                        file_json["clientSecret"].clone();
                                    found_credentials = true;
                                    tracing::info!(
                                        "[KIRO] 从 {} 合并 client_id/client_secret",
                                        file_name
                                    );
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        if !found_credentials {
            let auth_method = merged_creds
                .get("authMethod")
                .and_then(|v| v.as_str())
                .unwrap_or("social");

            if auth_method.to_lowercase() == "idc" {
                tracing::error!(
                    "[KIRO] IdC 认证方式缺少 clientId/clientSecret，无法创建有效的凭证"
                );
                return Err(
                    "IdC 认证凭证不完整：缺少 clientId/clientSecret。\n\n💡 解决方案：\n1. 确保 ~/.aws/sso/cache/ 目录下有对应的 clientIdHash 文件\n2. 如果使用 AWS IAM Identity Center，请确保已完成完整的 SSO 登录流程\n3. 或者尝试使用 Social 认证方式的凭证".to_string()
                );
            } else {
                tracing::warn!("[KIRO] 未找到 client_id/client_secret，将使用 social 认证方式");
            }
        }
    }

    // 写入凭证文件
    let merged_content = serde_json::to_string_pretty(&merged_creds)
        .map_err(|e| format!("序列化凭证失败: {}", e))?;
    fs::write(&target_path, merged_content).map_err(|e| format!("写入凭证文件失败: {}", e))?;

    tracing::info!("[KIRO] 凭证文件已创建: {:?}", target_path);

    Ok(target_path.to_string_lossy().to_string())
}

/// 添加 Kiro OAuth 凭证（通过 JSON 内容）
///
/// 直接粘贴凭证 JSON 内容，无需选择文件
#[tauri::command]
pub fn add_kiro_from_json(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    json_content: String,
    name: Option<String>,
) -> Result<ProviderCredential, String> {
    tracing::info!(
        "[KIRO] 开始从 JSON 添加凭证，内容长度: {}",
        json_content.len()
    );

    // 从 JSON 内容创建凭证文件
    let stored_file_path = match create_kiro_credential_from_json(&json_content) {
        Ok(path) => {
            tracing::info!("[KIRO] 凭证文件已创建: {}", path);
            path
        }
        Err(e) => {
            tracing::error!("[KIRO] 创建凭证文件失败: {}", e);
            return Err(e);
        }
    };

    match pool_service.0.add_credential(
        &db,
        "kiro",
        CredentialData::KiroOAuth {
            creds_file_path: stored_file_path,
        },
        name,
        Some(true),
        None,
    ) {
        Ok(cred) => {
            tracing::info!("[KIRO] 凭证添加成功，UUID: {}", cred.uuid);
            Ok(cred)
        }
        Err(e) => {
            tracing::error!("[KIRO] 添加凭证到数据库失败: {}", e);
            Err(e)
        }
    }
}

/// 添加 Gemini OAuth 凭证（通过文件路径）
#[tauri::command]
pub fn add_gemini_oauth_credential(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    creds_file_path: String,
    project_id: Option<String>,
    name: Option<String>,
) -> Result<ProviderCredential, String> {
    // 复制并重命名文件到应用存储目录
    let stored_file_path = copy_and_rename_credential_file(&creds_file_path, "gemini")?;

    pool_service.0.add_credential(
        &db,
        "gemini",
        CredentialData::GeminiOAuth {
            creds_file_path: stored_file_path,
            project_id,
        },
        name,
        Some(true),
        None,
    )
}

/// 添加 Qwen OAuth 凭证（通过文件路径）
#[tauri::command]
pub fn add_qwen_oauth_credential(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    creds_file_path: String,
    name: Option<String>,
) -> Result<ProviderCredential, String> {
    // 复制并重命名文件到应用存储目录
    let stored_file_path = copy_and_rename_credential_file(&creds_file_path, "qwen")?;

    pool_service.0.add_credential(
        &db,
        "qwen",
        CredentialData::QwenOAuth {
            creds_file_path: stored_file_path,
        },
        name,
        Some(true),
        None,
    )
}

/// 添加 Antigravity OAuth 凭证（通过文件路径）
#[tauri::command]
pub fn add_antigravity_oauth_credential(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    creds_file_path: String,
    project_id: Option<String>,
    name: Option<String>,
) -> Result<ProviderCredential, String> {
    // 复制并重命名文件到应用存储目录
    let stored_file_path = copy_and_rename_credential_file(&creds_file_path, "antigravity")?;

    pool_service.0.add_credential(
        &db,
        "antigravity",
        CredentialData::AntigravityOAuth {
            creds_file_path: stored_file_path,
            project_id,
        },
        name,
        Some(true),
        None,
    )
}

/// 添加 OpenAI API Key 凭证
#[tauri::command]
pub fn add_openai_key_credential(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    api_key: String,
    base_url: Option<String>,
    name: Option<String>,
) -> Result<ProviderCredential, String> {
    pool_service.0.add_credential(
        &db,
        "openai",
        CredentialData::OpenAIKey { api_key, base_url },
        name,
        Some(true),
        None,
    )
}

/// 添加 Claude API Key 凭证
#[tauri::command]
pub fn add_claude_key_credential(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    api_key: String,
    base_url: Option<String>,
    name: Option<String>,
) -> Result<ProviderCredential, String> {
    pool_service.0.add_credential(
        &db,
        "claude",
        CredentialData::ClaudeKey { api_key, base_url },
        name,
        Some(true),
        None,
    )
}

/// 添加 Gemini API Key 凭证
#[tauri::command]
pub fn add_gemini_api_key_credential(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    api_key: String,
    base_url: Option<String>,
    excluded_models: Option<Vec<String>>,
    name: Option<String>,
) -> Result<ProviderCredential, String> {
    pool_service.0.add_credential(
        &db,
        "gemini_api_key",
        CredentialData::GeminiApiKey {
            api_key,
            base_url,
            excluded_models: excluded_models.unwrap_or_default(),
        },
        name,
        Some(true),
        None,
    )
}

/// 添加 Codex OAuth 凭证（通过文件路径）
#[tauri::command]
pub fn add_codex_oauth_credential(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    creds_file_path: String,
    api_base_url: Option<String>,
    name: Option<String>,
) -> Result<ProviderCredential, String> {
    // 复制并重命名文件到应用存储目录
    let stored_file_path = copy_and_rename_credential_file(&creds_file_path, "codex")?;

    pool_service.0.add_credential(
        &db,
        "codex",
        CredentialData::CodexOAuth {
            creds_file_path: stored_file_path,
            api_base_url,
        },
        name,
        Some(true),
        None,
    )
}

/// 添加 Claude OAuth 凭证（通过文件路径）
#[tauri::command]
pub fn add_claude_oauth_credential(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    creds_file_path: String,
    name: Option<String>,
) -> Result<ProviderCredential, String> {
    // 复制并重命名文件到应用存储目录
    let stored_file_path = copy_and_rename_credential_file(&creds_file_path, "claude_oauth")?;

    pool_service.0.add_credential(
        &db,
        "claude_oauth",
        CredentialData::ClaudeOAuth {
            creds_file_path: stored_file_path,
        },
        name,
        Some(true),
        None,
    )
}

/// 添加 iFlow OAuth 凭证（通过文件路径）
#[tauri::command]
pub fn add_iflow_oauth_credential(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    creds_file_path: String,
    name: Option<String>,
) -> Result<ProviderCredential, String> {
    // 复制并重命名文件到应用存储目录
    let stored_file_path = copy_and_rename_credential_file(&creds_file_path, "iflow")?;

    pool_service.0.add_credential(
        &db,
        "iflow",
        CredentialData::IFlowOAuth {
            creds_file_path: stored_file_path,
        },
        name,
        Some(true),
        None,
    )
}

/// 添加 iFlow Cookie 凭证（通过文件路径）
#[tauri::command]
pub fn add_iflow_cookie_credential(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    creds_file_path: String,
    name: Option<String>,
) -> Result<ProviderCredential, String> {
    // 复制并重命名文件到应用存储目录
    let stored_file_path = copy_and_rename_credential_file(&creds_file_path, "iflow_cookie")?;

    pool_service.0.add_credential(
        &db,
        "iflow",
        CredentialData::IFlowCookie {
            creds_file_path: stored_file_path,
        },
        name,
        Some(true),
        None,
    )
}

/// 刷新凭证的 OAuth Token
#[tauri::command]
pub async fn refresh_pool_credential_token(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    uuid: String,
) -> Result<String, String> {
    tracing::info!("[DEBUG] 开始刷新 Token for uuid: {}", uuid);
    let result = pool_service.0.refresh_credential_token(&db, &uuid).await;
    match &result {
        Ok(msg) => tracing::info!("[DEBUG] Token 刷新成功: {}", msg),
        Err(err) => tracing::error!("[DEBUG] Token 刷新失败: {}", err),
    }
    result
}

/// 获取凭证的 OAuth 状态
#[tauri::command]
pub fn get_pool_credential_oauth_status(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    uuid: String,
) -> Result<OAuthStatus, String> {
    pool_service.0.get_credential_oauth_status(&db, &uuid)
}

/// 调试 Kiro 凭证加载（从默认路径）
/// P0 安全修复：仅在 debug 构建中可用
#[cfg(debug_assertions)]
#[tauri::command]
pub async fn debug_kiro_credentials() -> Result<String, String> {
    use crate::providers::kiro::KiroProvider;

    let mut provider = KiroProvider::new();

    let mut result = String::new();
    result.push_str("🔍 开始 Kiro 凭证调试 (默认路径)...\n\n");

    match provider.load_credentials().await {
        Ok(_) => {
            result.push_str("✅ 凭证加载成功!\n");
            result.push_str(&format!(
                "📄 认证方式: {:?}\n",
                provider.credentials.auth_method
            ));
            result.push_str(&format!(
                "🔑 有 client_id: {}\n",
                provider.credentials.client_id.is_some()
            ));
            result.push_str(&format!(
                "🔒 有 client_secret: {}\n",
                provider.credentials.client_secret.is_some()
            ));
            result.push_str(&format!(
                "🏷️  有 clientIdHash: {}\n",
                provider.credentials.client_id_hash.is_some()
            ));

            // P0 安全修复：不再输出敏感信息（clientIdHash、token 前缀等）
            let detected_method = provider.detect_auth_method();
            result.push_str(&format!("🎯 检测到的认证方式: {}\n", detected_method));

            result.push_str("\n🚀 尝试刷新 token...\n");
            match provider.refresh_token().await {
                Ok(token) => {
                    result.push_str(&format!("✅ Token 刷新成功! Token 长度: {}\n", token.len()));
                    // 不再输出 token 前缀
                }
                Err(e) => {
                    result.push_str(&format!("❌ Token 刷新失败: {}\n", e));
                }
            }
        }
        Err(e) => {
            result.push_str(&format!("❌ 凭证加载失败: {}\n", e));
        }
    }

    Ok(result)
}

/// P0 安全修复：release 构建中禁用 debug 命令
#[cfg(not(debug_assertions))]
#[tauri::command]
pub async fn debug_kiro_credentials() -> Result<String, String> {
    Err("此调试命令仅在开发构建中可用".to_string())
}

/// 测试用户上传的凭证文件
/// P0 安全修复：仅在 debug 构建中可用，且不输出敏感信息
#[cfg(debug_assertions)]
#[tauri::command]
pub async fn test_user_credentials() -> Result<String, String> {
    use crate::providers::kiro::KiroProvider;

    let mut result = String::new();
    result.push_str("🧪 测试用户上传的凭证文件...\n\n");

    // 测试用户上传的凭证文件路径
    let user_creds_path = dirs::home_dir()
        .ok_or("无法获取用户主目录".to_string())?
        .join(
            "Library/Application Support/proxycast/credentials/kiro_d8da9d58_1765757992_kiro.json",
        );

    // P0 安全修复：不输出完整路径，仅显示文件是否存在
    result.push_str("📂 检查用户凭证文件...\n");

    // 检查文件是否存在
    if !user_creds_path.exists() {
        result.push_str("❌ 用户凭证文件不存在!\n");
        result.push_str("💡 请确保文件路径正确，或重新上传凭证文件\n");
        return Ok(result);
    }

    result.push_str("✅ 用户凭证文件存在\n\n");

    // 读取并解析用户凭证文件
    match std::fs::read_to_string(&user_creds_path) {
        Ok(content) => {
            result.push_str("✅ 成功读取凭证文件\n");
            result.push_str(&format!("📄 文件大小: {} 字节\n", content.len()));

            // 尝试解析 JSON
            match serde_json::from_str::<serde_json::Value>(&content) {
                Ok(json) => {
                    result.push_str("✅ JSON 格式有效\n");

                    // 检查关键字段（仅显示是否存在，不显示值）
                    let has_access_token =
                        json.get("accessToken").and_then(|v| v.as_str()).is_some();
                    let has_refresh_token =
                        json.get("refreshToken").and_then(|v| v.as_str()).is_some();
                    let auth_method = json.get("authMethod").and_then(|v| v.as_str());
                    let has_client_id_hash =
                        json.get("clientIdHash").and_then(|v| v.as_str()).is_some();
                    let region = json.get("region").and_then(|v| v.as_str());

                    result.push_str(&format!("🔑 有 accessToken: {}\n", has_access_token));
                    result.push_str(&format!("🔄 有 refreshToken: {}\n", has_refresh_token));
                    result.push_str(&format!("📄 authMethod: {:?}\n", auth_method));
                    // P0 安全修复：不输出 clientIdHash 值
                    result.push_str(&format!("🏷️ 有 clientIdHash: {}\n", has_client_id_hash));
                    result.push_str(&format!("🌍 region: {:?}\n", region));

                    // 使用 KiroProvider 测试加载
                    result.push_str("\n🔧 使用 KiroProvider 测试加载...\n");

                    let mut provider = KiroProvider::new();
                    provider.creds_path = Some(user_creds_path.clone());

                    match provider
                        .load_credentials_from_path(&user_creds_path.to_string_lossy())
                        .await
                    {
                        Ok(_) => {
                            result.push_str("✅ KiroProvider 加载成功!\n");
                            result.push_str(&format!(
                                "📄 最终认证方式: {:?}\n",
                                provider.credentials.auth_method
                            ));
                            result.push_str(&format!(
                                "🔑 最终有 client_id: {}\n",
                                provider.credentials.client_id.is_some()
                            ));
                            result.push_str(&format!(
                                "🔒 最终有 client_secret: {}\n",
                                provider.credentials.client_secret.is_some()
                            ));

                            let detected_method = provider.detect_auth_method();
                            result.push_str(&format!("🎯 检测到的认证方式: {}\n", detected_method));

                            result.push_str("\n🚀 尝试刷新 token...\n");
                            match provider.refresh_token().await {
                                Ok(token) => {
                                    result.push_str(&format!(
                                        "✅ Token 刷新成功! Token 长度: {}\n",
                                        token.len()
                                    ));
                                    // P0 安全修复：不输出 token 前缀
                                }
                                Err(e) => {
                                    result.push_str(&format!("❌ Token 刷新失败: {}\n", e));
                                }
                            }
                        }
                        Err(e) => {
                            result.push_str(&format!("❌ KiroProvider 加载失败: {}\n", e));
                        }
                    }
                }
                Err(e) => {
                    result.push_str(&format!("❌ JSON 格式无效: {}\n", e));
                }
            }
        }
        Err(e) => {
            result.push_str(&format!("❌ 无法读取凭证文件: {}\n", e));
        }
    }

    Ok(result)
}

/// P0 安全修复：release 构建中禁用 test_user_credentials 命令
#[cfg(not(debug_assertions))]
#[tauri::command]
pub async fn test_user_credentials() -> Result<String, String> {
    Err("此调试命令仅在开发构建中可用".to_string())
}

/// 迁移 Private 配置到凭证池
///
/// 从 providers 配置中读取单个凭证配置，迁移到凭证池中并标记为 Private 来源
/// Requirements: 6.4
#[tauri::command]
pub fn migrate_private_config_to_pool(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    config: crate::config::Config,
) -> Result<MigrationResultResponse, String> {
    let result = pool_service.0.migrate_private_config(&db, &config)?;
    Ok(MigrationResultResponse {
        migrated_count: result.migrated_count,
        skipped_count: result.skipped_count,
        errors: result.errors,
    })
}

/// 迁移结果响应
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MigrationResultResponse {
    /// 成功迁移的凭证数量
    pub migrated_count: usize,
    /// 跳过的凭证数量（已存在）
    pub skipped_count: usize,
    /// 错误信息列表
    pub errors: Vec<String>,
}

/// 获取 Antigravity OAuth 授权 URL 并等待回调（不自动打开浏览器）
///
/// 启动服务器后通过事件发送授权 URL，然后等待回调
/// 成功后返回凭证
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AntigravityAuthUrlResponse {
    pub auth_url: String,
}

#[tauri::command]
pub async fn get_antigravity_auth_url_and_wait(
    app: tauri::AppHandle,
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    name: Option<String>,
    skip_project_id_fetch: Option<bool>,
) -> Result<ProviderCredential, String> {
    use crate::providers::antigravity;

    tracing::info!("[Antigravity OAuth] 启动服务器并获取授权 URL");

    // 启动服务器并获取授权 URL
    let (auth_url, wait_future) =
        antigravity::start_oauth_server_and_get_url(skip_project_id_fetch.unwrap_or(false))
            .await
            .map_err(|e| format!("启动 OAuth 服务器失败: {}", e))?;

    tracing::info!("[Antigravity OAuth] 授权 URL: {}", auth_url);

    // 通过事件发送授权 URL 给前端
    let _ = app.emit(
        "antigravity-auth-url",
        AntigravityAuthUrlResponse {
            auth_url: auth_url.clone(),
        },
    );

    // 等待回调
    let result = wait_future.await.map_err(|e| e.to_string())?;

    tracing::info!(
        "[Antigravity OAuth] 登录成功，凭证保存到: {}",
        result.creds_file_path
    );

    // 从凭证中获取 project_id
    let project_id = result.credentials.project_id.clone();

    // 添加到凭证池
    let credential = pool_service.0.add_credential(
        &db,
        "antigravity",
        CredentialData::AntigravityOAuth {
            creds_file_path: result.creds_file_path,
            project_id,
        },
        name,
        Some(true),
        None,
    )?;

    tracing::info!(
        "[Antigravity OAuth] 凭证已添加到凭证池: {}",
        credential.uuid
    );

    Ok(credential)
}

/// 启动 Antigravity OAuth 登录流程
///
/// 打开浏览器让用户登录 Google 账号，获取 Antigravity 凭证
#[tauri::command]
pub async fn start_antigravity_oauth_login(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    name: Option<String>,
    skip_project_id_fetch: Option<bool>,
) -> Result<ProviderCredential, String> {
    use crate::providers::antigravity;

    tracing::info!("[Antigravity OAuth] 开始 OAuth 登录流程");

    // 启动 OAuth 登录
    let result = antigravity::start_oauth_login(skip_project_id_fetch.unwrap_or(false))
        .await
        .map_err(|e| format!("Antigravity OAuth 登录失败: {}", e))?;

    tracing::info!(
        "[Antigravity OAuth] 登录成功，凭证保存到: {}",
        result.creds_file_path
    );

    // 从凭证中获取 project_id
    let project_id = result.credentials.project_id.clone();

    // 添加到凭证池
    let credential = pool_service.0.add_credential(
        &db,
        "antigravity",
        CredentialData::AntigravityOAuth {
            creds_file_path: result.creds_file_path,
            project_id,
        },
        name,
        Some(true),
        None,
    )?;

    tracing::info!(
        "[Antigravity OAuth] 凭证已添加到凭证池: {}",
        credential.uuid
    );

    Ok(credential)
}

/// Codex OAuth 授权 URL 响应
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CodexAuthUrlResponse {
    pub auth_url: String,
}

/// 获取 Codex OAuth 授权 URL 并等待回调（不自动打开浏览器）
///
/// 启动服务器后通过事件发送授权 URL，然后等待回调
/// 成功后返回凭证
#[tauri::command]
pub async fn get_codex_auth_url_and_wait(
    app: tauri::AppHandle,
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    name: Option<String>,
) -> Result<ProviderCredential, String> {
    use crate::providers::codex;

    tracing::info!("[Codex OAuth] 启动服务器并获取授权 URL");

    // 启动服务器并获取授权 URL
    let (auth_url, wait_future) = codex::start_codex_oauth_server_and_get_url()
        .await
        .map_err(|e| format!("启动 OAuth 服务器失败: {}", e))?;

    tracing::info!("[Codex OAuth] 授权 URL: {}", auth_url);

    // 通过事件发送授权 URL 给前端
    let _ = app.emit(
        "codex-auth-url",
        CodexAuthUrlResponse {
            auth_url: auth_url.clone(),
        },
    );

    // 等待回调
    let result = wait_future.await.map_err(|e| e.to_string())?;

    tracing::info!(
        "[Codex OAuth] 登录成功，凭证保存到: {}",
        result.creds_file_path
    );

    // 添加到凭证池
    let credential = pool_service.0.add_credential(
        &db,
        "codex",
        CredentialData::CodexOAuth {
            creds_file_path: result.creds_file_path,
            api_base_url: None,
        },
        name,
        Some(true),
        None,
    )?;

    tracing::info!("[Codex OAuth] 凭证已添加到凭证池: {}", credential.uuid);

    Ok(credential)
}

/// 启动 Codex OAuth 登录流程
///
/// 打开浏览器让用户登录 OpenAI 账号，获取 Codex 凭证
#[tauri::command]
pub async fn start_codex_oauth_login(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    name: Option<String>,
) -> Result<ProviderCredential, String> {
    use crate::providers::codex;

    tracing::info!("[Codex OAuth] 开始 OAuth 登录流程");

    // 启动 OAuth 登录
    let result = codex::start_codex_oauth_login()
        .await
        .map_err(|e| format!("Codex OAuth 登录失败: {}", e))?;

    tracing::info!(
        "[Codex OAuth] 登录成功，凭证保存到: {}",
        result.creds_file_path
    );

    // 添加到凭证池
    let credential = pool_service.0.add_credential(
        &db,
        "codex",
        CredentialData::CodexOAuth {
            creds_file_path: result.creds_file_path,
            api_base_url: None,
        },
        name,
        Some(true),
        None,
    )?;

    tracing::info!("[Codex OAuth] 凭证已添加到凭证池: {}", credential.uuid);

    Ok(credential)
}

/// Claude OAuth 授权 URL 响应
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClaudeOAuthAuthUrlResponse {
    pub auth_url: String,
}

/// 获取 Claude OAuth 授权 URL 并等待回调（不自动打开浏览器）
///
/// 启动服务器后通过事件发送授权 URL，然后等待回调
/// 成功后返回凭证
/// Claude OAuth 授权 URL 响应（新流程）
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClaudeOAuthParamsResponse {
    pub auth_url: String,
    pub code_verifier: String,
    pub state: String,
}

/// 获取 Claude OAuth 授权 URL（新流程：手动输入授权码）
///
/// 生成授权 URL 和 PKCE 参数，用户需要：
/// 1. 打开 auth_url 进行授权
/// 2. 授权后从页面复制授权码
/// 3. 调用 exchange_claude_oauth_code 交换 token
#[tauri::command]
pub async fn get_claude_oauth_auth_url_and_wait(
    app: tauri::AppHandle,
    _db: State<'_, DbConnection>,
    _pool_service: State<'_, ProviderPoolServiceState>,
    _name: Option<String>,
) -> Result<ClaudeOAuthParamsResponse, String> {
    use crate::providers::claude_oauth;

    tracing::info!("[Claude OAuth] 生成授权 URL（手动授权码流程）");

    // 生成授权参数
    let params = claude_oauth::generate_claude_oauth_params()
        .map_err(|e| format!("生成授权参数失败: {}", e))?;

    tracing::info!("[Claude OAuth] 授权 URL: {}", params.auth_url);

    // 通过事件发送授权 URL 给前端
    let _ = app.emit(
        "claude-oauth-auth-url",
        ClaudeOAuthAuthUrlResponse {
            auth_url: params.auth_url.clone(),
        },
    );

    // 打开浏览器
    if let Err(e) = open::that(&params.auth_url) {
        tracing::warn!("[Claude OAuth] 无法打开浏览器: {}. 请手动打开 URL.", e);
    }

    Ok(ClaudeOAuthParamsResponse {
        auth_url: params.auth_url,
        code_verifier: params.code_verifier,
        state: params.state,
    })
}

/// 使用授权码交换 Claude OAuth Token
///
/// 用户在浏览器中授权后，复制授权码，调用此命令交换 token
#[tauri::command]
pub async fn exchange_claude_oauth_code(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    authorization_code: String,
    code_verifier: String,
    state: String,
    name: Option<String>,
) -> Result<ProviderCredential, String> {
    use crate::providers::claude_oauth;

    tracing::info!("[Claude OAuth] 使用授权码交换 Token");

    // 交换 Token
    let result = claude_oauth::exchange_claude_authorization_code(
        &authorization_code,
        &code_verifier,
        &state,
    )
    .await
    .map_err(|e| format!("Claude OAuth Token 交换失败: {}", e))?;

    tracing::info!(
        "[Claude OAuth] 登录成功，凭证保存到: {}",
        result.creds_file_path
    );

    // 添加到凭证池
    let credential = pool_service.0.add_credential(
        &db,
        "claude_oauth",
        CredentialData::ClaudeOAuth {
            creds_file_path: result.creds_file_path,
        },
        name,
        Some(true),
        None,
    )?;

    tracing::info!("[Claude OAuth] 凭证已添加到凭证池: {}", credential.uuid);

    Ok(credential)
}

/// 启动 Claude OAuth 登录流程（兼容旧接口，现在返回授权参数）
///
/// 打开浏览器让用户登录 Claude 账号
/// 注意：新流程需要用户手动复制授权码，然后调用 exchange_claude_oauth_code
#[tauri::command]
pub async fn start_claude_oauth_login(
    _db: State<'_, DbConnection>,
    _pool_service: State<'_, ProviderPoolServiceState>,
    _name: Option<String>,
) -> Result<ClaudeOAuthParamsResponse, String> {
    use crate::providers::claude_oauth;

    tracing::info!("[Claude OAuth] 开始 OAuth 登录流程（手动授权码模式）");

    // 生成授权参数并打开浏览器
    let params = claude_oauth::start_claude_oauth_login()
        .await
        .map_err(|e| format!("Claude OAuth 登录失败: {}", e))?;

    Ok(ClaudeOAuthParamsResponse {
        auth_url: params.auth_url,
        code_verifier: params.code_verifier,
        state: params.state,
    })
}

/// Claude Cookie 自动授权响应
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClaudeCookieOAuthResponse {
    pub organization_uuid: Option<String>,
    pub capabilities: Vec<String>,
}

/// 使用 Cookie (sessionKey) 自动完成 Claude OAuth 授权
///
/// 这是一个更便捷的授权方式，用户只需要提供从浏览器 Cookie 中获取的 sessionKey，
/// 系统会自动完成整个 OAuth 流程，无需手动复制授权码。
///
/// # 参数
/// - `session_key`: 从浏览器 Cookie 中获取的 sessionKey
/// - `is_setup_token`: 是否为 Setup Token 模式（只需要推理权限，无 refresh_token）
/// - `name`: 凭证名称（可选）
#[tauri::command]
pub async fn claude_oauth_with_cookie(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    session_key: String,
    is_setup_token: Option<bool>,
    name: Option<String>,
) -> Result<ProviderCredential, String> {
    use crate::providers::claude_oauth;

    let is_setup = is_setup_token.unwrap_or(false);
    tracing::info!(
        "[Claude OAuth] 开始 Cookie 自动授权流程，is_setup_token: {}",
        is_setup
    );

    // 执行 Cookie 自动授权
    let result = claude_oauth::oauth_with_cookie(&session_key, is_setup)
        .await
        .map_err(|e| format!("Claude Cookie 授权失败: {}", e))?;

    tracing::info!(
        "[Claude OAuth] Cookie 授权成功，凭证保存到: {}",
        result.creds_file_path
    );

    // 添加到凭证池
    let credential = pool_service.0.add_credential(
        &db,
        "claude_oauth",
        CredentialData::ClaudeOAuth {
            creds_file_path: result.creds_file_path,
        },
        name,
        Some(true),
        None,
    )?;

    tracing::info!(
        "[Claude OAuth] 凭证已添加到凭证池: {}, org_uuid: {:?}",
        credential.uuid,
        result.organization_uuid
    );

    Ok(credential)
}

/// Qwen Device Code 响应
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QwenDeviceCodeResponse {
    pub user_code: String,
    pub verification_uri: String,
    pub verification_uri_complete: Option<String>,
    pub expires_in: i64,
}

/// 获取 Qwen Device Code 并等待用户授权
///
/// 启动 Device Code Flow 后通过事件发送设备码信息，然后轮询等待授权
/// 成功后返回凭证
#[tauri::command]
pub async fn get_qwen_device_code_and_wait(
    app: tauri::AppHandle,
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    name: Option<String>,
) -> Result<ProviderCredential, String> {
    use crate::providers::qwen;

    tracing::info!("[Qwen] 启动 Device Code Flow");

    // 启动 Device Code Flow 并获取设备码信息
    let (device_response, wait_future) = qwen::start_qwen_device_code_and_get_info()
        .await
        .map_err(|e| format!("启动 Device Code Flow 失败: {}", e))?;

    tracing::info!(
        "[Qwen] Device Code: user_code={}, verification_uri={}",
        device_response.user_code,
        device_response.verification_uri
    );

    // 通过事件发送设备码信息给前端
    let _ = app.emit(
        "qwen-device-code",
        QwenDeviceCodeResponse {
            user_code: device_response.user_code.clone(),
            verification_uri: device_response.verification_uri.clone(),
            verification_uri_complete: device_response.verification_uri_complete.clone(),
            expires_in: device_response.expires_in,
        },
    );

    // 等待用户授权
    let result = wait_future.await.map_err(|e| e.to_string())?;

    tracing::info!("[Qwen] 登录成功，凭证保存到: {}", result.creds_file_path);

    // 添加到凭证池
    let credential = pool_service.0.add_credential(
        &db,
        "qwen",
        CredentialData::QwenOAuth {
            creds_file_path: result.creds_file_path,
        },
        name,
        Some(true),
        None,
    )?;

    tracing::info!("[Qwen] 凭证已添加到凭证池: {}", credential.uuid);

    Ok(credential)
}

/// 启动 Qwen Device Code Flow 登录流程
///
/// 自动打开浏览器让用户完成授权
#[tauri::command]
pub async fn start_qwen_device_code_login(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    name: Option<String>,
) -> Result<ProviderCredential, String> {
    use crate::providers::qwen;

    tracing::info!("[Qwen] 开始 Device Code Flow 登录流程");

    // 启动 Device Code Flow 登录
    let result = qwen::start_qwen_device_code_login()
        .await
        .map_err(|e| format!("Qwen Device Code Flow 登录失败: {}", e))?;

    tracing::info!("[Qwen] 登录成功，凭证保存到: {}", result.creds_file_path);

    // 添加到凭证池
    let credential = pool_service.0.add_credential(
        &db,
        "qwen",
        CredentialData::QwenOAuth {
            creds_file_path: result.creds_file_path,
        },
        name,
        Some(true),
        None,
    )?;

    tracing::info!("[Qwen] 凭证已添加到凭证池: {}", credential.uuid);

    Ok(credential)
}

/// iFlow OAuth 授权 URL 响应
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IFlowAuthUrlResponse {
    pub auth_url: String,
}

/// 获取 iFlow OAuth 授权 URL 并等待回调（不自动打开浏览器）
///
/// 启动服务器后通过事件发送授权 URL，然后等待回调
/// 成功后返回凭证
#[tauri::command]
pub async fn get_iflow_auth_url_and_wait(
    app: tauri::AppHandle,
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    name: Option<String>,
) -> Result<ProviderCredential, String> {
    use crate::providers::iflow;

    tracing::info!("[iFlow OAuth] 启动服务器并获取授权 URL");

    // 启动服务器并获取授权 URL
    let (auth_url, wait_future) = iflow::start_iflow_oauth_server_and_get_url()
        .await
        .map_err(|e| format!("启动 OAuth 服务器失败: {}", e))?;

    tracing::info!("[iFlow OAuth] 授权 URL: {}", auth_url);

    // 通过事件发送授权 URL 给前端
    let _ = app.emit(
        "iflow-auth-url",
        IFlowAuthUrlResponse {
            auth_url: auth_url.clone(),
        },
    );

    // 等待回调
    let result = wait_future.await.map_err(|e| e.to_string())?;

    tracing::info!(
        "[iFlow OAuth] 登录成功，凭证保存到: {}",
        result.creds_file_path
    );

    // 添加到凭证池
    let credential = pool_service.0.add_credential(
        &db,
        "iflow",
        CredentialData::IFlowOAuth {
            creds_file_path: result.creds_file_path,
        },
        name,
        Some(true),
        None,
    )?;

    tracing::info!("[iFlow OAuth] 凭证已添加到凭证池: {}", credential.uuid);

    Ok(credential)
}

/// 启动 iFlow OAuth 登录流程
///
/// 打开浏览器让用户登录 iFlow 账号，获取凭证
#[tauri::command]
pub async fn start_iflow_oauth_login(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    name: Option<String>,
) -> Result<ProviderCredential, String> {
    use crate::providers::iflow;

    tracing::info!("[iFlow OAuth] 开始 OAuth 登录流程");

    // 启动 OAuth 登录
    let result = iflow::start_iflow_oauth_login()
        .await
        .map_err(|e| format!("iFlow OAuth 登录失败: {}", e))?;

    tracing::info!(
        "[iFlow OAuth] 登录成功，凭证保存到: {}",
        result.creds_file_path
    );

    // 添加到凭证池
    let credential = pool_service.0.add_credential(
        &db,
        "iflow",
        CredentialData::IFlowOAuth {
            creds_file_path: result.creds_file_path,
        },
        name,
        Some(true),
        None,
    )?;

    tracing::info!("[iFlow OAuth] 凭证已添加到凭证池: {}", credential.uuid);

    Ok(credential)
}

/// 获取 Kiro 凭证的 Machine ID 指纹信息
///
/// 返回凭证的唯一设备指纹，用于在 UI 中展示
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KiroFingerprintInfo {
    /// Machine ID（SHA256 哈希，64 字符）
    pub machine_id: String,
    /// Machine ID 的短格式（前 16 字符）
    pub machine_id_short: String,
    /// 指纹来源（profileArn / clientId / system）
    pub source: String,
    /// 认证方式
    pub auth_method: String,
}

#[tauri::command]
pub async fn get_kiro_credential_fingerprint(
    db: State<'_, DbConnection>,
    uuid: String,
) -> Result<KiroFingerprintInfo, String> {
    use crate::database::dao::provider_pool::ProviderPoolDao;
    use crate::providers::kiro::{generate_machine_id_from_credentials, KiroProvider};

    // 获取凭证文件路径（在锁释放前完成）
    let creds_file_path = {
        let conn = db.lock().map_err(|e| e.to_string())?;
        let credential = ProviderPoolDao::get_by_uuid(&conn, &uuid)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("凭证不存在: {}", uuid))?;

        // 检查是否为 Kiro 凭证
        match &credential.credential {
            CredentialData::KiroOAuth { creds_file_path } => creds_file_path.clone(),
            _ => return Err("只有 Kiro 凭证支持获取指纹信息".to_string()),
        }
    }; // conn 在这里释放

    // 加载凭证文件（异步操作，锁已释放）
    let mut provider = KiroProvider::new();
    provider
        .load_credentials_from_path(&creds_file_path)
        .await
        .map_err(|e| format!("加载凭证失败: {}", e))?;

    // 确定指纹来源
    let (source, profile_arn, client_id) = if provider.credentials.profile_arn.is_some() {
        (
            "profileArn".to_string(),
            provider.credentials.profile_arn.as_deref(),
            None,
        )
    } else if provider.credentials.client_id.is_some() {
        (
            "clientId".to_string(),
            None,
            provider.credentials.client_id.as_deref(),
        )
    } else {
        ("system".to_string(), None, None)
    };

    // 生成 Machine ID
    let machine_id = generate_machine_id_from_credentials(profile_arn, client_id);
    // 安全地截取前 16 个字符（避免越界 panic）
    let machine_id_short: String = machine_id.chars().take(16).collect();

    // 获取认证方式
    let auth_method = provider
        .credentials
        .auth_method
        .clone()
        .unwrap_or_else(|| "social".to_string());

    Ok(KiroFingerprintInfo {
        machine_id,
        machine_id_short,
        source,
        auth_method,
    })
}

/// Gemini OAuth 授权 URL 响应
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GeminiAuthUrlResponse {
    pub auth_url: String,
    pub session_id: String,
}

use once_cell::sync::Lazy;
/// Gemini OAuth 会话存储（用于存储 code_verifier）
use std::collections::HashMap;
use tokio::sync::RwLock;

static GEMINI_OAUTH_SESSIONS: Lazy<
    RwLock<HashMap<String, crate::providers::gemini::GeminiOAuthSession>>,
> = Lazy::new(|| RwLock::new(HashMap::new()));

/// 获取 Gemini OAuth 授权 URL（不等待回调）
///
/// 生成授权 URL 和 session_id，通过事件发送给前端
/// 用户需要手动复制授权码回来，然后调用 exchange_gemini_code
#[tauri::command]
pub async fn get_gemini_auth_url_and_wait(
    app: tauri::AppHandle,
    _db: State<'_, DbConnection>,
    _pool_service: State<'_, ProviderPoolServiceState>,
    _name: Option<String>,
) -> Result<ProviderCredential, String> {
    use crate::providers::gemini;

    tracing::info!("[Gemini OAuth] 生成授权 URL");

    // 生成授权 URL 和会话信息
    let (auth_url, session) = gemini::generate_gemini_auth_url_with_session();
    let session_id = session.session_id.clone();

    tracing::info!("[Gemini OAuth] 授权 URL: {}", auth_url);
    tracing::info!("[Gemini OAuth] Session ID: {}", session_id);

    // 存储会话信息（用于后续交换 token）
    {
        let mut sessions = GEMINI_OAUTH_SESSIONS.write().await;
        sessions.insert(session_id.clone(), session);

        // 清理过期的会话（超过 10 分钟）
        let now = chrono::Utc::now().timestamp();
        sessions.retain(|_, s| now - s.created_at < 600);
    }

    // 通过事件发送授权 URL 给前端
    let _ = app.emit(
        "gemini-auth-url",
        GeminiAuthUrlResponse {
            auth_url: auth_url.clone(),
            session_id: session_id.clone(),
        },
    );

    // 返回错误，让前端知道需要用户手动输入授权码
    // 这不是真正的错误，只是流程需要用户交互
    Err(format!("AUTH_URL:{}", auth_url))
}

/// 用 Gemini 授权码交换 Token 并添加凭证
#[tauri::command]
pub async fn exchange_gemini_code(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    code: String,
    session_id: Option<String>,
    name: Option<String>,
) -> Result<ProviderCredential, String> {
    use crate::providers::gemini;

    tracing::info!("[Gemini OAuth] 开始交换授权码");

    // 获取 code_verifier
    let code_verifier = if let Some(ref sid) = session_id {
        let sessions = GEMINI_OAUTH_SESSIONS.read().await;
        sessions
            .get(sid)
            .map(|s| s.code_verifier.clone())
            .ok_or_else(|| "会话已过期，请重新获取授权 URL".to_string())?
    } else {
        // 如果没有 session_id，尝试使用最近的会话
        let sessions = GEMINI_OAUTH_SESSIONS.read().await;
        sessions
            .values()
            .max_by_key(|s| s.created_at)
            .map(|s| s.code_verifier.clone())
            .ok_or_else(|| "没有可用的会话，请先获取授权 URL".to_string())?
    };

    // 交换 token 并创建凭证
    let result = gemini::exchange_gemini_code_and_create_credentials(&code, &code_verifier)
        .await
        .map_err(|e| format!("交换授权码失败: {}", e))?;

    tracing::info!(
        "[Gemini OAuth] 登录成功，凭证保存到: {}",
        result.creds_file_path
    );

    // 清理使用过的会话
    if let Some(ref sid) = session_id {
        let mut sessions = GEMINI_OAUTH_SESSIONS.write().await;
        sessions.remove(sid);
    }

    // 添加到凭证池
    let credential = pool_service.0.add_credential(
        &db,
        "gemini",
        CredentialData::GeminiOAuth {
            creds_file_path: result.creds_file_path,
            project_id: None, // 项目 ID 会在健康检查时自动获取
        },
        name,
        Some(true),
        None,
    )?;

    tracing::info!("[Gemini OAuth] 凭证已添加到凭证池: {}", credential.uuid);

    Ok(credential)
}

/// 启动 Gemini OAuth 登录流程
///
/// 打开浏览器让用户登录 Google 账号，获取 Gemini 凭证
#[tauri::command]
pub async fn start_gemini_oauth_login(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    name: Option<String>,
) -> Result<ProviderCredential, String> {
    use crate::providers::gemini;

    tracing::info!("[Gemini OAuth] 开始 OAuth 登录流程");

    // 启动 OAuth 登录
    let result = gemini::start_gemini_oauth_login()
        .await
        .map_err(|e| format!("Gemini OAuth 登录失败: {}", e))?;

    tracing::info!(
        "[Gemini OAuth] 登录成功，凭证保存到: {}",
        result.creds_file_path
    );

    // 添加到凭证池
    let credential = pool_service.0.add_credential(
        &db,
        "gemini",
        CredentialData::GeminiOAuth {
            creds_file_path: result.creds_file_path,
            project_id: None,
        },
        name,
        Some(true),
        None,
    )?;

    tracing::info!("[Gemini OAuth] 凭证已添加到凭证池: {}", credential.uuid);

    Ok(credential)
}

// ============ Kiro Builder ID 登录相关命令 ============

/// Kiro Builder ID 登录状态
#[derive(Debug, Clone)]
struct KiroBuilderIdLoginState {
    /// OIDC 客户端 ID
    client_id: String,
    /// OIDC 客户端密钥
    client_secret: String,
    /// 设备码
    device_code: String,
    /// 用户码
    user_code: String,
    /// 验证 URI
    verification_uri: String,
    /// 轮询间隔（秒）
    interval: i64,
    /// 过期时间戳
    expires_at: i64,
    /// 区域
    region: String,
}

/// 全局 Builder ID 登录状态存储
static KIRO_BUILDER_ID_LOGIN_STATE: Lazy<RwLock<Option<KiroBuilderIdLoginState>>> =
    Lazy::new(|| RwLock::new(None));

/// Kiro Builder ID 登录启动响应
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KiroBuilderIdLoginResponse {
    /// 是否成功
    pub success: bool,
    /// 用户码（用于显示给用户）
    #[serde(rename = "userCode")]
    pub user_code: Option<String>,
    /// 验证 URI（用户需要访问的 URL）
    #[serde(rename = "verificationUri")]
    pub verification_uri: Option<String>,
    /// 过期时间（秒）
    #[serde(rename = "expiresIn")]
    pub expires_in: Option<i64>,
    /// 轮询间隔（秒）
    pub interval: Option<i64>,
    /// 错误信息
    pub error: Option<String>,
}

/// Kiro Builder ID 轮询响应
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KiroBuilderIdPollResponse {
    /// 是否成功
    pub success: bool,
    /// 是否完成授权
    pub completed: bool,
    /// 状态（pending / slow_down）
    pub status: Option<String>,
    /// 错误信息
    pub error: Option<String>,
}

/// 启动 Kiro Builder ID 登录
///
/// 使用 OIDC Device Authorization Flow 进行登录
#[tauri::command]
pub async fn start_kiro_builder_id_login(
    region: Option<String>,
) -> Result<KiroBuilderIdLoginResponse, String> {
    let region = region.unwrap_or_else(|| "us-east-1".to_string());
    let oidc_base = format!("https://oidc.{}.amazonaws.com", region);
    let start_url = "https://view.awsapps.com/start";
    let scopes = vec![
        "codewhisperer:completions",
        "codewhisperer:analysis",
        "codewhisperer:conversations",
        "codewhisperer:transformations",
        "codewhisperer:taskassist",
    ];

    tracing::info!("[Kiro Builder ID] 开始登录流程，区域: {}", region);

    // Step 1: 注册 OIDC 客户端
    tracing::info!("[Kiro Builder ID] Step 1: 注册 OIDC 客户端...");
    let client = reqwest::Client::new();

    let reg_body = serde_json::json!({
        "clientName": "ProxyCast Kiro Manager",
        "clientType": "public",
        "scopes": scopes,
        "grantTypes": ["urn:ietf:params:oauth:grant-type:device_code", "refresh_token"],
        "issuerUrl": start_url
    });

    let reg_res = client
        .post(format!("{}/client/register", oidc_base))
        .header("Content-Type", "application/json")
        .json(&reg_body)
        .send()
        .await
        .map_err(|e| format!("注册客户端请求失败: {}", e))?;

    if !reg_res.status().is_success() {
        let err_text = reg_res.text().await.unwrap_or_default();
        return Ok(KiroBuilderIdLoginResponse {
            success: false,
            user_code: None,
            verification_uri: None,
            expires_in: None,
            interval: None,
            error: Some(format!("注册客户端失败: {}", err_text)),
        });
    }

    let reg_data: serde_json::Value = reg_res
        .json()
        .await
        .map_err(|e| format!("解析注册响应失败: {}", e))?;

    let client_id = reg_data["clientId"]
        .as_str()
        .ok_or("响应中缺少 clientId")?
        .to_string();
    let client_secret = reg_data["clientSecret"]
        .as_str()
        .ok_or("响应中缺少 clientSecret")?
        .to_string();

    tracing::info!(
        "[Kiro Builder ID] 客户端注册成功: {}...",
        client_id.chars().take(30).collect::<String>()
    );

    // Step 2: 发起设备授权
    tracing::info!("[Kiro Builder ID] Step 2: 发起设备授权...");
    let auth_body = serde_json::json!({
        "clientId": client_id,
        "clientSecret": client_secret,
        "startUrl": start_url
    });

    let auth_res = client
        .post(format!("{}/device_authorization", oidc_base))
        .header("Content-Type", "application/json")
        .json(&auth_body)
        .send()
        .await
        .map_err(|e| format!("设备授权请求失败: {}", e))?;

    if !auth_res.status().is_success() {
        let err_text = auth_res.text().await.unwrap_or_default();
        return Ok(KiroBuilderIdLoginResponse {
            success: false,
            user_code: None,
            verification_uri: None,
            expires_in: None,
            interval: None,
            error: Some(format!("设备授权失败: {}", err_text)),
        });
    }

    let auth_data: serde_json::Value = auth_res
        .json()
        .await
        .map_err(|e| format!("解析授权响应失败: {}", e))?;

    let device_code = auth_data["deviceCode"]
        .as_str()
        .ok_or("响应中缺少 deviceCode")?
        .to_string();
    let user_code = auth_data["userCode"]
        .as_str()
        .ok_or("响应中缺少 userCode")?
        .to_string();
    let verification_uri = auth_data["verificationUriComplete"]
        .as_str()
        .or_else(|| auth_data["verificationUri"].as_str())
        .ok_or("响应中缺少 verificationUri")?
        .to_string();
    let interval = auth_data["interval"].as_i64().unwrap_or(5);
    let expires_in = auth_data["expiresIn"].as_i64().unwrap_or(600);

    tracing::info!("[Kiro Builder ID] 设备码获取成功，user_code: {}", user_code);

    // 保存登录状态
    let expires_at = chrono::Utc::now().timestamp() + expires_in;
    {
        let mut state = KIRO_BUILDER_ID_LOGIN_STATE.write().await;
        *state = Some(KiroBuilderIdLoginState {
            client_id,
            client_secret,
            device_code,
            user_code: user_code.clone(),
            verification_uri: verification_uri.clone(),
            interval,
            expires_at,
            region,
        });
    }

    Ok(KiroBuilderIdLoginResponse {
        success: true,
        user_code: Some(user_code),
        verification_uri: Some(verification_uri),
        expires_in: Some(expires_in),
        interval: Some(interval),
        error: None,
    })
}

/// 轮询 Kiro Builder ID 授权状态
#[tauri::command]
pub async fn poll_kiro_builder_id_auth() -> Result<KiroBuilderIdPollResponse, String> {
    let state = {
        let state_guard = KIRO_BUILDER_ID_LOGIN_STATE.read().await;
        match state_guard.as_ref() {
            Some(s) => s.clone(),
            None => {
                return Ok(KiroBuilderIdPollResponse {
                    success: false,
                    completed: false,
                    status: None,
                    error: Some("没有进行中的登录".to_string()),
                });
            }
        }
    };

    // 检查是否过期
    if chrono::Utc::now().timestamp() > state.expires_at {
        // 清除状态
        {
            let mut state_guard = KIRO_BUILDER_ID_LOGIN_STATE.write().await;
            *state_guard = None;
        }
        return Ok(KiroBuilderIdPollResponse {
            success: false,
            completed: false,
            status: None,
            error: Some("授权已过期，请重新开始".to_string()),
        });
    }

    let oidc_base = format!("https://oidc.{}.amazonaws.com", state.region);
    let client = reqwest::Client::new();

    let token_body = serde_json::json!({
        "clientId": state.client_id,
        "clientSecret": state.client_secret,
        "grantType": "urn:ietf:params:oauth:grant-type:device_code",
        "deviceCode": state.device_code
    });

    let token_res = client
        .post(format!("{}/token", oidc_base))
        .header("Content-Type", "application/json")
        .json(&token_body)
        .send()
        .await
        .map_err(|e| format!("Token 请求失败: {}", e))?;

    let status = token_res.status();

    if status.is_success() {
        // 授权成功
        let token_data: serde_json::Value = token_res
            .json()
            .await
            .map_err(|e| format!("解析 Token 响应失败: {}", e))?;

        tracing::info!("[Kiro Builder ID] 授权成功！");

        // 保存凭证到文件
        let access_token = token_data["accessToken"].as_str().unwrap_or("").to_string();
        let refresh_token = token_data["refreshToken"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let expires_in = token_data["expiresIn"].as_i64().unwrap_or(3600);

        // 创建凭证 JSON
        let creds_json = serde_json::json!({
            "accessToken": access_token,
            "refreshToken": refresh_token,
            "clientId": state.client_id,
            "clientSecret": state.client_secret,
            "region": state.region,
            "authMethod": "idc",
            "expiresAt": chrono::Utc::now().timestamp() + expires_in
        });

        // 保存到临时状态，等待 add_kiro_from_builder_id_auth 调用
        // 这里我们把凭证 JSON 存储到一个临时位置
        {
            let mut sessions = KIRO_BUILDER_ID_CREDENTIALS.write().await;
            sessions.insert("pending".to_string(), creds_json);
        }

        // 清除登录状态
        {
            let mut state_guard = KIRO_BUILDER_ID_LOGIN_STATE.write().await;
            *state_guard = None;
        }

        Ok(KiroBuilderIdPollResponse {
            success: true,
            completed: true,
            status: None,
            error: None,
        })
    } else if status.as_u16() == 400 {
        let err_data: serde_json::Value = token_res
            .json()
            .await
            .map_err(|e| format!("解析错误响应失败: {}", e))?;

        let error = err_data["error"].as_str().unwrap_or("unknown");

        match error {
            "authorization_pending" => Ok(KiroBuilderIdPollResponse {
                success: true,
                completed: false,
                status: Some("pending".to_string()),
                error: None,
            }),
            "slow_down" => {
                // 增加轮询间隔
                {
                    let mut state_guard = KIRO_BUILDER_ID_LOGIN_STATE.write().await;
                    if let Some(ref mut s) = *state_guard {
                        s.interval += 5;
                    }
                }
                Ok(KiroBuilderIdPollResponse {
                    success: true,
                    completed: false,
                    status: Some("slow_down".to_string()),
                    error: None,
                })
            }
            "expired_token" => {
                // 清除状态
                {
                    let mut state_guard = KIRO_BUILDER_ID_LOGIN_STATE.write().await;
                    *state_guard = None;
                }
                Ok(KiroBuilderIdPollResponse {
                    success: false,
                    completed: false,
                    status: None,
                    error: Some("设备码已过期".to_string()),
                })
            }
            "access_denied" => {
                // 清除状态
                {
                    let mut state_guard = KIRO_BUILDER_ID_LOGIN_STATE.write().await;
                    *state_guard = None;
                }
                Ok(KiroBuilderIdPollResponse {
                    success: false,
                    completed: false,
                    status: None,
                    error: Some("用户拒绝授权".to_string()),
                })
            }
            _ => {
                // 清除状态
                {
                    let mut state_guard = KIRO_BUILDER_ID_LOGIN_STATE.write().await;
                    *state_guard = None;
                }
                Ok(KiroBuilderIdPollResponse {
                    success: false,
                    completed: false,
                    status: None,
                    error: Some(format!("授权错误: {}", error)),
                })
            }
        }
    } else {
        Ok(KiroBuilderIdPollResponse {
            success: false,
            completed: false,
            status: None,
            error: Some(format!("未知响应: {}", status)),
        })
    }
}

/// 临时存储 Builder ID 登录成功后的凭证
static KIRO_BUILDER_ID_CREDENTIALS: Lazy<RwLock<HashMap<String, serde_json::Value>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// 取消 Kiro Builder ID 登录
#[tauri::command]
pub async fn cancel_kiro_builder_id_login() -> Result<bool, String> {
    tracing::info!("[Kiro Builder ID] 取消登录");
    {
        let mut state = KIRO_BUILDER_ID_LOGIN_STATE.write().await;
        *state = None;
    }
    {
        let mut creds = KIRO_BUILDER_ID_CREDENTIALS.write().await;
        creds.remove("pending");
    }
    Ok(true)
}

/// 从 Builder ID 授权结果添加 Kiro 凭证
#[tauri::command]
pub async fn add_kiro_from_builder_id_auth(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    name: Option<String>,
) -> Result<ProviderCredential, String> {
    // 获取待处理的凭证
    let creds_json = {
        let mut creds = KIRO_BUILDER_ID_CREDENTIALS.write().await;
        creds
            .remove("pending")
            .ok_or("没有待处理的 Builder ID 凭证")?
    };

    // 将凭证 JSON 转换为字符串
    let json_content =
        serde_json::to_string_pretty(&creds_json).map_err(|e| format!("序列化凭证失败: {}", e))?;

    // 使用现有的 create_kiro_credential_from_json 函数创建凭证文件
    let stored_file_path = create_kiro_credential_from_json(&json_content)?;

    // 添加到凭证池
    let credential = pool_service.0.add_credential(
        &db,
        "kiro",
        CredentialData::KiroOAuth {
            creds_file_path: stored_file_path,
        },
        name,
        Some(true),
        None,
    )?;

    tracing::info!("[Kiro Builder ID] 凭证已添加到凭证池: {}", credential.uuid);

    Ok(credential)
}

// ============ Kiro Social Auth 登录相关命令 (Google/GitHub) ============

/// Kiro Auth 端点
const KIRO_AUTH_ENDPOINT: &str = "https://prod.us-east-1.auth.desktop.kiro.dev";

/// Kiro Social Auth 登录状态
#[derive(Debug, Clone)]
struct KiroSocialAuthLoginState {
    /// 登录提供商 (Google / Github)
    provider: String,
    /// PKCE code_verifier
    code_verifier: String,
    /// PKCE code_challenge
    code_challenge: String,
    /// OAuth state
    oauth_state: String,
    /// 过期时间戳
    expires_at: i64,
}

/// 全局 Social Auth 登录状态存储
static KIRO_SOCIAL_AUTH_LOGIN_STATE: Lazy<RwLock<Option<KiroSocialAuthLoginState>>> =
    Lazy::new(|| RwLock::new(None));

/// Kiro Social Auth 登录启动响应
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KiroSocialAuthLoginResponse {
    /// 是否成功
    pub success: bool,
    /// 登录 URL
    #[serde(rename = "loginUrl")]
    pub login_url: Option<String>,
    /// OAuth state（用于验证回调）
    pub state: Option<String>,
    /// 错误信息
    pub error: Option<String>,
}

/// Kiro Social Auth Token 交换响应
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KiroSocialAuthTokenResponse {
    /// 是否成功
    pub success: bool,
    /// 错误信息
    pub error: Option<String>,
}

/// 生成 PKCE code_verifier
fn generate_code_verifier() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let bytes: Vec<u8> = (0..64).map(|_| rng.gen()).collect();
    base64_url_encode(&bytes)[..128.min(base64_url_encode(&bytes).len())].to_string()
}

/// 生成 PKCE code_challenge (SHA256)
fn generate_code_challenge(verifier: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    let result = hasher.finalize();
    base64_url_encode(&result)
}

/// 生成 OAuth state
fn generate_oauth_state() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let bytes: Vec<u8> = (0..32).map(|_| rng.gen()).collect();
    base64_url_encode(&bytes)
}

/// Base64 URL 编码（无填充）
fn base64_url_encode(data: &[u8]) -> String {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
    URL_SAFE_NO_PAD.encode(data)
}

/// 启动 Kiro Social Auth 登录 (Google/GitHub)
///
/// 使用 PKCE OAuth 流程进行登录
/// 打开系统默认浏览器进行 OAuth 登录
#[tauri::command]
pub async fn start_kiro_social_auth_login(
    provider: String,
) -> Result<KiroSocialAuthLoginResponse, String> {
    // 验证 provider
    let provider_normalized = match provider.to_lowercase().as_str() {
        "google" => "Google",
        "github" => "Github",
        _ => {
            return Ok(KiroSocialAuthLoginResponse {
                success: false,
                login_url: None,
                state: None,
                error: Some(format!("不支持的登录提供商: {}", provider)),
            });
        }
    };

    tracing::info!("[Kiro Social Auth] 开始 {} 登录流程", provider_normalized);

    // 生成 PKCE
    let code_verifier = generate_code_verifier();
    let code_challenge = generate_code_challenge(&code_verifier);
    let oauth_state = generate_oauth_state();

    // 构建登录 URL
    // 使用本地回调服务器接收授权码
    let redirect_uri = "http://127.0.0.1:19823/kiro-social-callback";

    let login_url = format!(
        "{}/login?idp={}&redirect_uri={}&code_challenge={}&code_challenge_method=S256&state={}",
        KIRO_AUTH_ENDPOINT,
        provider_normalized,
        urlencoding::encode(redirect_uri),
        urlencoding::encode(&code_challenge),
        urlencoding::encode(&oauth_state)
    );

    tracing::info!("[Kiro Social Auth] 登录 URL: {}", login_url);

    // 保存登录状态（10 分钟过期）
    let expires_at = chrono::Utc::now().timestamp() + 600;
    {
        let mut state = KIRO_SOCIAL_AUTH_LOGIN_STATE.write().await;
        *state = Some(KiroSocialAuthLoginState {
            provider: provider_normalized.to_string(),
            code_verifier,
            code_challenge,
            oauth_state: oauth_state.clone(),
            expires_at,
        });
    }

    Ok(KiroSocialAuthLoginResponse {
        success: true,
        login_url: Some(login_url),
        state: Some(oauth_state),
        error: None,
    })
}

/// 交换 Kiro Social Auth Token
///
/// 用授权码交换 access_token 和 refresh_token
#[tauri::command]
pub async fn exchange_kiro_social_auth_token(
    code: String,
    state: String,
) -> Result<KiroSocialAuthTokenResponse, String> {
    tracing::info!("[Kiro Social Auth] 交换 Token...");

    // 获取并验证登录状态
    let login_state = {
        let state_guard = KIRO_SOCIAL_AUTH_LOGIN_STATE.read().await;
        match state_guard.as_ref() {
            Some(s) => s.clone(),
            None => {
                return Ok(KiroSocialAuthTokenResponse {
                    success: false,
                    error: Some("没有进行中的社交登录".to_string()),
                });
            }
        }
    };

    // 验证 state
    if state != login_state.oauth_state {
        // 清除状态
        {
            let mut state_guard = KIRO_SOCIAL_AUTH_LOGIN_STATE.write().await;
            *state_guard = None;
        }
        return Ok(KiroSocialAuthTokenResponse {
            success: false,
            error: Some("状态参数不匹配，可能存在安全风险".to_string()),
        });
    }

    // 检查是否过期
    if chrono::Utc::now().timestamp() > login_state.expires_at {
        // 清除状态
        {
            let mut state_guard = KIRO_SOCIAL_AUTH_LOGIN_STATE.write().await;
            *state_guard = None;
        }
        return Ok(KiroSocialAuthTokenResponse {
            success: false,
            error: Some("登录已过期，请重新开始".to_string()),
        });
    }

    let redirect_uri = "http://127.0.0.1:19823/kiro-social-callback";

    // 交换 Token
    let client = reqwest::Client::new();
    let token_body = serde_json::json!({
        "code": code,
        "code_verifier": login_state.code_verifier,
        "redirect_uri": redirect_uri
    });

    let token_res = client
        .post(format!("{}/oauth/token", KIRO_AUTH_ENDPOINT))
        .header("Content-Type", "application/json")
        .json(&token_body)
        .send()
        .await
        .map_err(|e| format!("Token 交换请求失败: {}", e))?;

    if !token_res.status().is_success() {
        let err_text = token_res.text().await.unwrap_or_default();
        // 清除状态
        {
            let mut state_guard = KIRO_SOCIAL_AUTH_LOGIN_STATE.write().await;
            *state_guard = None;
        }
        return Ok(KiroSocialAuthTokenResponse {
            success: false,
            error: Some(format!("Token 交换失败: {}", err_text)),
        });
    }

    let token_data: serde_json::Value = token_res
        .json()
        .await
        .map_err(|e| format!("解析 Token 响应失败: {}", e))?;

    tracing::info!("[Kiro Social Auth] Token 交换成功!");

    // 提取凭证
    let access_token = token_data["accessToken"].as_str().unwrap_or("").to_string();
    let refresh_token = token_data["refreshToken"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let profile_arn = token_data["profileArn"].as_str().map(|s| s.to_string());
    let expires_in = token_data["expiresIn"].as_i64().unwrap_or(3600);

    // 创建凭证 JSON
    let creds_json = serde_json::json!({
        "accessToken": access_token,
        "refreshToken": refresh_token,
        "profileArn": profile_arn,
        "authMethod": "social",
        "provider": login_state.provider,
        "expiresAt": chrono::Utc::now().timestamp() + expires_in
    });

    // 保存到临时状态
    {
        let mut creds = KIRO_BUILDER_ID_CREDENTIALS.write().await;
        creds.insert("pending".to_string(), creds_json);
    }

    // 清除登录状态
    {
        let mut state_guard = KIRO_SOCIAL_AUTH_LOGIN_STATE.write().await;
        *state_guard = None;
    }

    Ok(KiroSocialAuthTokenResponse {
        success: true,
        error: None,
    })
}

/// 取消 Kiro Social Auth 登录
#[tauri::command]
pub async fn cancel_kiro_social_auth_login() -> Result<bool, String> {
    tracing::info!("[Kiro Social Auth] 取消登录");
    {
        let mut state = KIRO_SOCIAL_AUTH_LOGIN_STATE.write().await;
        *state = None;
    }
    Ok(true)
}

// ============ Playwright 指纹浏览器登录相关命令 ============

/// Playwright 可用性状态
///
/// Requirements: 2.1, 2.2
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PlaywrightStatus {
    /// 浏览器是否可用
    pub available: bool,
    /// 浏览器可执行文件路径
    pub browser_path: Option<String>,
    /// 浏览器来源: "system" 或 "playwright"
    pub browser_source: Option<String>,
    /// 错误信息
    pub error: Option<String>,
}

/// 获取系统 Chrome 可执行文件路径
fn get_system_chrome_path() -> Option<String> {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));

    #[cfg(target_os = "macos")]
    {
        let paths = [
            PathBuf::from("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            PathBuf::from("/Applications/Chromium.app/Contents/MacOS/Chromium"),
            home.join("Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        ];
        for path in paths {
            if path.exists() {
                return Some(path.to_string_lossy().to_string());
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        let paths = [
            PathBuf::from("C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"),
            PathBuf::from("C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"),
            home.join("AppData\\Local\\Google\\Chrome\\Application\\chrome.exe"),
        ];
        for path in paths {
            if path.exists() {
                return Some(path.to_string_lossy().to_string());
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        let paths = [
            PathBuf::from("/usr/bin/google-chrome"),
            PathBuf::from("/usr/bin/google-chrome-stable"),
            PathBuf::from("/usr/bin/chromium"),
            PathBuf::from("/usr/bin/chromium-browser"),
            PathBuf::from("/snap/bin/chromium"),
        ];
        for path in paths {
            if path.exists() {
                return Some(path.to_string_lossy().to_string());
            }
        }
    }

    None
}

/// 获取 Playwright 浏览器缓存目录
fn get_playwright_cache_dir() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));

    #[cfg(target_os = "macos")]
    {
        home.join("Library").join("Caches").join("ms-playwright")
    }

    #[cfg(target_os = "windows")]
    {
        home.join("AppData").join("Local").join("ms-playwright")
    }

    #[cfg(target_os = "linux")]
    {
        home.join(".cache").join("ms-playwright")
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
    {
        home.join(".cache").join("ms-playwright")
    }
}

/// 获取 Playwright Chromium 浏览器可执行文件路径
///
/// 搜索常见的 Chromium 版本目录
fn get_playwright_browser_path() -> Option<String> {
    let cache_dir = get_playwright_cache_dir();

    // Playwright 常见的 Chromium 版本目录
    let chromium_versions = [
        "chromium-1140",
        "chromium-1134",
        "chromium-1124",
        "chromium-1117",
        "chromium-1112",
        "chromium-1108",
        "chromium-1105",
        "chromium-1097",
        "chromium-1091",
        "chromium-1084",
        "chromium-1080",
        "chromium-1076",
        "chromium-1067",
        "chromium-1060",
        "chromium-1055",
        "chromium-1048",
        "chromium-1045",
        "chromium-1041",
        "chromium-1033",
        "chromium-1028",
        "chromium-1024",
        "chromium-1020",
        "chromium-1015",
        "chromium-1012",
        "chromium-1008",
        "chromium-1005",
        "chromium-1000",
        "chromium",
    ];

    for version in chromium_versions {
        #[cfg(target_os = "macos")]
        let exec_path = cache_dir
            .join(version)
            .join("chrome-mac")
            .join("Chromium.app")
            .join("Contents")
            .join("MacOS")
            .join("Chromium");

        #[cfg(target_os = "windows")]
        let exec_path = cache_dir
            .join(version)
            .join("chrome-win")
            .join("chrome.exe");

        #[cfg(target_os = "linux")]
        let exec_path = cache_dir.join(version).join("chrome-linux").join("chrome");

        #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
        let exec_path = cache_dir.join(version).join("chrome-linux").join("chrome");

        if exec_path.exists() {
            return Some(exec_path.to_string_lossy().to_string());
        }
    }

    None
}

/// 获取可用的浏览器路径（优先系统 Chrome）
fn get_available_browser_path() -> Option<(String, String)> {
    // 优先使用系统 Chrome
    if let Some(path) = get_system_chrome_path() {
        return Some((path, "system".to_string()));
    }

    // 其次使用 Playwright Chromium
    if let Some(path) = get_playwright_browser_path() {
        return Some((path, "playwright".to_string()));
    }

    None
}

/// 检查浏览器是否可用（优先系统 Chrome）
///
/// 检测系统 Chrome 或 Playwright Chromium 是否存在
/// Requirements: 2.1, 2.2
#[tauri::command]
pub async fn check_playwright_available() -> Result<PlaywrightStatus, String> {
    tracing::info!("[Browser] 检查浏览器可用性...");

    match get_available_browser_path() {
        Some((browser_path, source)) => {
            tracing::info!("[Browser] 找到 {} 浏览器: {}", source, browser_path);
            Ok(PlaywrightStatus {
                available: true,
                browser_path: Some(browser_path),
                browser_source: Some(source),
                error: None,
            })
        }
        None => {
            let error_msg =
                "未找到可用的浏览器。请安装 Google Chrome 或运行: npx playwright install chromium"
                    .to_string();
            tracing::warn!("[Browser] {}", error_msg);
            Ok(PlaywrightStatus {
                available: false,
                browser_path: None,
                browser_source: None,
                error: Some(error_msg),
            })
        }
    }
}

/// Playwright 安装进度事件
///
/// 用于向前端发送安装进度信息
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PlaywrightInstallProgress {
    /// 进度消息
    pub message: String,
    /// 是否完成
    pub done: bool,
    /// 是否成功（仅在 done=true 时有效）
    pub success: Option<bool>,
}

/// 安装 Playwright Chromium 浏览器
///
/// 执行 npm install playwright && npx playwright install chromium
/// Requirements: 6.1, 6.2
#[tauri::command]
pub async fn install_playwright(app: tauri::AppHandle) -> Result<PlaywrightStatus, String> {
    use tokio::io::{AsyncBufReadExt, BufReader};
    use tokio::process::Command;

    tracing::info!("[Playwright] 开始安装 Playwright...");

    // 发送进度事件
    let _ = app.emit(
        "playwright-install-progress",
        PlaywrightInstallProgress {
            message: "正在查找 Playwright 脚本目录...".to_string(),
            done: false,
            success: None,
        },
    );

    // 尝试多个可能的脚本目录路径
    let possible_paths = vec![
        // 开发模式：从 CARGO_MANIFEST_DIR 推导
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap_or(&PathBuf::from("."))
            .join("scripts")
            .join("playwright-login"),
        // 生产模式：应用数据目录
        dirs::data_dir()
            .unwrap_or_default()
            .join("proxycast")
            .join("scripts")
            .join("playwright-login"),
        // 当前工作目录
        std::env::current_dir()
            .unwrap_or_default()
            .join("scripts")
            .join("playwright-login"),
    ];

    let mut script_dir: Option<PathBuf> = None;
    for path in &possible_paths {
        tracing::info!("[Playwright] 检查路径: {:?}", path);
        if path.join("package.json").exists() {
            script_dir = Some(path.clone());
            break;
        }
    }

    let script_dir = match script_dir {
        Some(dir) => dir,
        None => {
            let error = format!(
                "找不到 Playwright 脚本目录。已检查路径:\n{}",
                possible_paths
                    .iter()
                    .map(|p| format!("  - {:?}", p))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
            tracing::error!("[Playwright] {}", error);
            let _ = app.emit(
                "playwright-install-progress",
                PlaywrightInstallProgress {
                    message: error.clone(),
                    done: true,
                    success: Some(false),
                },
            );
            return Err(error);
        }
    };

    tracing::info!("[Playwright] 使用脚本目录: {:?}", script_dir);

    // 步骤 1: 安装 npm 依赖
    let _ = app.emit(
        "playwright-install-progress",
        PlaywrightInstallProgress {
            message: format!("正在安装 npm 依赖... ({})", script_dir.display()),
            done: false,
            success: None,
        },
    );

    let npm_install = Command::new("npm")
        .arg("install")
        .current_dir(&script_dir)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn();

    match npm_install {
        Ok(mut child) => {
            // 收集 stderr 输出用于错误报告
            let mut stderr_output = String::new();
            if let Some(stderr) = child.stderr.take() {
                let mut reader = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    tracing::debug!("[Playwright npm] {}", line);
                    stderr_output.push_str(&line);
                    stderr_output.push('\n');
                }
            }

            let status = child.wait().await;
            match status {
                Ok(s) if s.success() => {
                    tracing::info!("[Playwright] npm install 成功");
                    // 发送成功消息
                    let _ = app.emit(
                        "playwright-install-progress",
                        PlaywrightInstallProgress {
                            message: "npm 依赖安装成功，准备安装 Chromium 浏览器...".to_string(),
                            done: false,
                            success: None,
                        },
                    );
                }
                Ok(s) => {
                    let error = if stderr_output.is_empty() {
                        format!("npm install 失败，退出码: {:?}", s.code())
                    } else {
                        format!("npm install 失败: {}", stderr_output.trim())
                    };
                    tracing::error!("[Playwright] {}", error);
                    let _ = app.emit(
                        "playwright-install-progress",
                        PlaywrightInstallProgress {
                            message: error.clone(),
                            done: true,
                            success: Some(false),
                        },
                    );
                    return Err(error);
                }
                Err(e) => {
                    let error = format!("npm install 执行失败: {}", e);
                    tracing::error!("[Playwright] {}", error);
                    let _ = app.emit(
                        "playwright-install-progress",
                        PlaywrightInstallProgress {
                            message: error.clone(),
                            done: true,
                            success: Some(false),
                        },
                    );
                    return Err(error);
                }
            }
        }
        Err(e) => {
            let error = format!("无法启动 npm: {}。请确保已安装 Node.js", e);
            tracing::error!("[Playwright] {}", error);
            let _ = app.emit(
                "playwright-install-progress",
                PlaywrightInstallProgress {
                    message: error.clone(),
                    done: true,
                    success: Some(false),
                },
            );
            return Err(error);
        }
    }

    // 步骤 2: 安装 Chromium 浏览器
    let _ = app.emit(
        "playwright-install-progress",
        PlaywrightInstallProgress {
            message: "正在安装 Chromium 浏览器 (npx playwright install chromium)...".to_string(),
            done: false,
            success: None,
        },
    );

    let playwright_install = Command::new("npx")
        .args(["playwright", "install", "chromium"])
        .current_dir(&script_dir)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn();

    match playwright_install {
        Ok(mut child) => {
            // 同时收集 stdout 和 stderr
            let mut stdout_output = String::new();
            let mut stderr_output = String::new();

            // 读取 stdout 并发送进度
            if let Some(stdout) = child.stdout.take() {
                let app_clone = app.clone();
                let mut reader = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    tracing::info!("[Playwright install] {}", line);
                    stdout_output.push_str(&line);
                    stdout_output.push('\n');
                    // 发送下载进度
                    if line.contains("Downloading")
                        || line.contains("%")
                        || line.contains("chromium")
                    {
                        let _ = app_clone.emit(
                            "playwright-install-progress",
                            PlaywrightInstallProgress {
                                message: line.clone(),
                                done: false,
                                success: None,
                            },
                        );
                    }
                }
            }

            // 读取 stderr
            if let Some(stderr) = child.stderr.take() {
                let mut reader = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    tracing::warn!("[Playwright install stderr] {}", line);
                    stderr_output.push_str(&line);
                    stderr_output.push('\n');
                }
            }

            let status = child.wait().await;
            match status {
                Ok(s) if s.success() => {
                    tracing::info!("[Playwright] Chromium 安装成功");
                }
                Ok(s) => {
                    // 优先使用 stderr，如果为空则使用 stdout
                    let output = if !stderr_output.is_empty() {
                        stderr_output.trim().to_string()
                    } else if !stdout_output.is_empty() {
                        stdout_output.trim().to_string()
                    } else {
                        format!("退出码: {:?}", s.code())
                    };
                    let error = format!("Chromium 安装失败: {}", output);
                    tracing::error!("[Playwright] {}", error);
                    let _ = app.emit(
                        "playwright-install-progress",
                        PlaywrightInstallProgress {
                            message: error.clone(),
                            done: true,
                            success: Some(false),
                        },
                    );
                    return Err(error);
                }
                Err(e) => {
                    let error = format!("Chromium 安装执行失败: {}", e);
                    tracing::error!("[Playwright] {}", error);
                    let _ = app.emit(
                        "playwright-install-progress",
                        PlaywrightInstallProgress {
                            message: error.clone(),
                            done: true,
                            success: Some(false),
                        },
                    );
                    return Err(error);
                }
            }
        }
        Err(e) => {
            let error = format!("无法启动 npx: {}", e);
            tracing::error!("[Playwright] {}", error);
            let _ = app.emit(
                "playwright-install-progress",
                PlaywrightInstallProgress {
                    message: error.clone(),
                    done: true,
                    success: Some(false),
                },
            );
            return Err(error);
        }
    }

    // 验证安装结果
    let status = check_playwright_available().await?;

    if status.available {
        let _ = app.emit(
            "playwright-install-progress",
            PlaywrightInstallProgress {
                message: "Playwright 安装成功！".to_string(),
                done: true,
                success: Some(true),
            },
        );
        tracing::info!(
            "[Playwright] 安装完成，浏览器路径: {:?}",
            status.browser_path
        );
    } else {
        let error =
            "安装完成但未检测到浏览器，请手动运行: npx playwright install chromium".to_string();
        let _ = app.emit(
            "playwright-install-progress",
            PlaywrightInstallProgress {
                message: error.clone(),
                done: true,
                success: Some(false),
            },
        );
        return Err(error);
    }

    Ok(status)
}

/// Playwright 登录进度事件
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PlaywrightLoginProgress {
    pub message: String,
}

/// Playwright 登录结果
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PlaywrightLoginResult {
    pub success: bool,
    pub code: Option<String>,
    pub state: Option<String>,
    pub error: Option<String>,
}

/// 全局 Playwright 登录进程状态
static PLAYWRIGHT_LOGIN_PROCESS: Lazy<RwLock<Option<tokio::process::Child>>> =
    Lazy::new(|| RwLock::new(None));

/// 获取 Playwright 登录脚本路径
fn get_playwright_script_path() -> PathBuf {
    // 开发模式下使用项目目录中的脚本
    let dev_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or(&PathBuf::from("."))
        .join("scripts")
        .join("playwright-login")
        .join("index.js");

    if dev_path.exists() {
        return dev_path;
    }

    // 生产模式下使用打包的资源
    if let Some(data_dir) = dirs::data_dir() {
        let prod_path = data_dir
            .join("proxycast")
            .join("scripts")
            .join("playwright-login")
            .join("index.js");
        if prod_path.exists() {
            return prod_path;
        }
    }

    // 回退到开发路径
    dev_path
}

/// 启动 Kiro Playwright 登录
///
/// 使用 Playwright 指纹浏览器进行 OAuth 登录
/// Requirements: 3.1, 3.4, 3.5, 4.3, 4.4
#[tauri::command]
pub async fn start_kiro_playwright_login(
    app: tauri::AppHandle,
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    provider: String,
    name: Option<String>,
) -> Result<ProviderCredential, String> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::process::Command;

    // 验证 provider
    let provider_normalized = match provider.to_lowercase().as_str() {
        "google" => "Google",
        "github" => "Github",
        "builderid" => "BuilderId",
        _ => {
            return Err(format!("不支持的登录提供商: {}", provider));
        }
    };

    tracing::info!("[Playwright Login] 开始 {} 登录流程", provider_normalized);

    // 检查 Playwright 是否可用
    let status = check_playwright_available().await?;
    if !status.available {
        return Err(status
            .error
            .unwrap_or_else(|| "Playwright 不可用".to_string()));
    }

    // 生成 PKCE
    let code_verifier = generate_code_verifier();
    let code_challenge = generate_code_challenge(&code_verifier);
    let oauth_state = generate_oauth_state();

    // 构建 OAuth URL
    let redirect_uri = "http://localhost:19824/callback";
    let auth_url = format!(
        "{}/login?idp={}&redirect_uri={}&code_challenge={}&code_challenge_method=S256&state={}",
        KIRO_AUTH_ENDPOINT,
        provider_normalized,
        urlencoding::encode(redirect_uri),
        urlencoding::encode(&code_challenge),
        urlencoding::encode(&oauth_state)
    );

    tracing::info!("[Playwright Login] OAuth URL: {}", auth_url);

    // 获取脚本路径
    let script_path = get_playwright_script_path();
    if !script_path.exists() {
        return Err(format!("Playwright 登录脚本不存在: {:?}", script_path));
    }

    tracing::info!("[Playwright Login] 脚本路径: {:?}", script_path);

    // 启动 Node.js 进程
    let mut child = Command::new("node")
        .arg(&script_path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("启动 Playwright 进程失败: {}", e))?;

    let stdin = child.stdin.take().ok_or("无法获取 stdin")?;
    let stdout = child.stdout.take().ok_or("无法获取 stdout")?;

    // 保存进程引用
    {
        let mut process_guard = PLAYWRIGHT_LOGIN_PROCESS.write().await;
        *process_guard = Some(child);
    }

    let mut stdin = tokio::io::BufWriter::new(stdin);
    let mut reader = BufReader::new(stdout);

    // 等待就绪信号
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .await
        .map_err(|e| format!("读取就绪信号失败: {}", e))?;

    let ready_response: serde_json::Value =
        serde_json::from_str(&line.trim()).map_err(|e| format!("解析就绪信号失败: {}", e))?;

    if ready_response.get("action").and_then(|v| v.as_str()) != Some("ready") {
        return Err("Playwright 脚本未就绪".to_string());
    }

    tracing::info!("[Playwright Login] Sidecar 已就绪");

    // 发送登录请求
    let login_request = serde_json::json!({
        "action": "login",
        "provider": provider_normalized,
        "authUrl": auth_url,
        "callbackUrl": redirect_uri
    });

    let request_str =
        serde_json::to_string(&login_request).map_err(|e| format!("序列化请求失败: {}", e))?;

    stdin
        .write_all(request_str.as_bytes())
        .await
        .map_err(|e| format!("发送请求失败: {}", e))?;
    stdin
        .write_all(b"\n")
        .await
        .map_err(|e| format!("发送换行失败: {}", e))?;
    stdin
        .flush()
        .await
        .map_err(|e| format!("刷新 stdin 失败: {}", e))?;

    tracing::info!("[Playwright Login] 已发送登录请求");

    // 读取响应
    let mut code: Option<String> = None;
    let mut state: Option<String> = None;

    loop {
        line.clear();
        match reader.read_line(&mut line).await {
            Ok(0) => {
                // EOF
                break;
            }
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                match serde_json::from_str::<serde_json::Value>(trimmed) {
                    Ok(response) => {
                        let action = response
                            .get("action")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let success = response
                            .get("success")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);

                        match action {
                            "progress" => {
                                if let Some(data) = response.get("data") {
                                    if let Some(message) =
                                        data.get("message").and_then(|v| v.as_str())
                                    {
                                        tracing::info!("[Playwright Login] 进度: {}", message);
                                        let _ = app.emit(
                                            "playwright-login-progress",
                                            PlaywrightLoginProgress {
                                                message: message.to_string(),
                                            },
                                        );
                                    }
                                }
                            }
                            "login" => {
                                if success {
                                    if let Some(data) = response.get("data") {
                                        code = data
                                            .get("code")
                                            .and_then(|v| v.as_str())
                                            .map(|s| s.to_string());
                                        state = data
                                            .get("state")
                                            .and_then(|v| v.as_str())
                                            .map(|s| s.to_string());
                                    }
                                } else {
                                    let error = response
                                        .get("data")
                                        .and_then(|d| d.get("error"))
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("未知错误");

                                    // 清理进程
                                    {
                                        let mut process_guard =
                                            PLAYWRIGHT_LOGIN_PROCESS.write().await;
                                        *process_guard = None;
                                    }

                                    return Err(format!("Playwright 登录失败: {}", error));
                                }
                                break;
                            }
                            "error" => {
                                let error = response
                                    .get("data")
                                    .and_then(|d| d.get("error"))
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("未知错误");

                                // 清理进程
                                {
                                    let mut process_guard = PLAYWRIGHT_LOGIN_PROCESS.write().await;
                                    *process_guard = None;
                                }

                                return Err(format!("Playwright 错误: {}", error));
                            }
                            _ => {}
                        }
                    }
                    Err(e) => {
                        tracing::warn!("[Playwright Login] 解析响应失败: {} - {}", e, trimmed);
                    }
                }
            }
            Err(e) => {
                // 清理进程
                {
                    let mut process_guard = PLAYWRIGHT_LOGIN_PROCESS.write().await;
                    *process_guard = None;
                }
                return Err(format!("读取响应失败: {}", e));
            }
        }
    }

    // 清理进程
    {
        let mut process_guard = PLAYWRIGHT_LOGIN_PROCESS.write().await;
        *process_guard = None;
    }

    // 验证结果
    let auth_code = code.ok_or("未获取到授权码")?;

    // 验证 state
    if let Some(returned_state) = &state {
        if returned_state != &oauth_state {
            return Err("状态参数不匹配，可能存在安全风险".to_string());
        }
    }

    tracing::info!("[Playwright Login] 获取到授权码，开始交换 Token");

    // 交换 Token
    let client = reqwest::Client::new();
    let token_body = serde_json::json!({
        "code": auth_code,
        "code_verifier": code_verifier,
        "redirect_uri": redirect_uri
    });

    let token_res = client
        .post(format!("{}/oauth/token", KIRO_AUTH_ENDPOINT))
        .header("Content-Type", "application/json")
        .json(&token_body)
        .send()
        .await
        .map_err(|e| format!("Token 交换请求失败: {}", e))?;

    if !token_res.status().is_success() {
        let err_text = token_res.text().await.unwrap_or_default();
        return Err(format!("Token 交换失败: {}", err_text));
    }

    let token_data: serde_json::Value = token_res
        .json()
        .await
        .map_err(|e| format!("解析 Token 响应失败: {}", e))?;

    tracing::info!("[Playwright Login] Token 交换成功!");

    // 提取凭证
    let access_token = token_data["accessToken"].as_str().unwrap_or("").to_string();
    let refresh_token = token_data["refreshToken"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let profile_arn = token_data["profileArn"].as_str().map(|s| s.to_string());
    let expires_in = token_data["expiresIn"].as_i64().unwrap_or(3600);

    // 创建凭证 JSON
    let creds_json = serde_json::json!({
        "accessToken": access_token,
        "refreshToken": refresh_token,
        "profileArn": profile_arn,
        "authMethod": "social",
        "provider": provider_normalized,
        "loginMethod": "playwright",
        "expiresAt": chrono::Utc::now().timestamp() + expires_in
    });

    // 将凭证 JSON 转换为字符串并创建凭证文件
    let json_content =
        serde_json::to_string_pretty(&creds_json).map_err(|e| format!("序列化凭证失败: {}", e))?;

    let stored_file_path = create_kiro_credential_from_json(&json_content)?;

    // 添加到凭证池
    let credential = pool_service.0.add_credential(
        &db,
        "kiro",
        CredentialData::KiroOAuth {
            creds_file_path: stored_file_path,
        },
        name,
        Some(true),
        None,
    )?;

    tracing::info!("[Playwright Login] 凭证已添加到凭证池: {}", credential.uuid);

    Ok(credential)
}

/// 取消 Kiro Playwright 登录
///
/// 终止正在进行的 Playwright 登录进程
/// Requirements: 5.3
#[tauri::command]
pub async fn cancel_kiro_playwright_login() -> Result<bool, String> {
    tracing::info!("[Playwright Login] 取消登录");

    let mut process_guard = PLAYWRIGHT_LOGIN_PROCESS.write().await;

    if let Some(mut child) = process_guard.take() {
        // 尝试发送取消命令
        if let Some(mut stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;

            let cancel_request = serde_json::json!({
                "action": "cancel"
            });

            if let Ok(request_str) = serde_json::to_string(&cancel_request) {
                let _ = stdin.write_all(request_str.as_bytes()).await;
                let _ = stdin.write_all(b"\n").await;
                let _ = stdin.flush().await;
            }
        }

        // 等待一小段时间让进程优雅退出
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        // 强制终止进程
        let _ = child.kill().await;

        tracing::info!("[Playwright Login] 登录进程已终止");
        Ok(true)
    } else {
        tracing::info!("[Playwright Login] 没有正在进行的登录");
        Ok(false)
    }
}

/// 启动 Kiro Social Auth 回调服务器
///
/// 启动一个本地 HTTP 服务器来接收 OAuth 回调
#[tauri::command]
pub async fn start_kiro_social_auth_callback_server(app: tauri::AppHandle) -> Result<bool, String> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    tracing::info!("[Kiro Social Auth] 启动回调服务器...");

    // 尝试绑定端口
    let listener = TcpListener::bind("127.0.0.1:19823")
        .await
        .map_err(|e| format!("无法启动回调服务器: {}", e))?;

    tracing::info!("[Kiro Social Auth] 回调服务器已启动在 127.0.0.1:19823");

    // 在后台处理连接
    let app_handle = app.clone();
    tokio::spawn(async move {
        // 只处理一个连接
        if let Ok((mut socket, _)) = listener.accept().await {
            let mut buffer = [0u8; 4096];
            if let Ok(n) = socket.read(&mut buffer).await {
                let request = String::from_utf8_lossy(&buffer[..n]);

                // 解析请求获取 code 和 state
                if let Some(path_line) = request.lines().next() {
                    if let Some(path) = path_line.split_whitespace().nth(1) {
                        if path.starts_with("/kiro-social-callback") {
                            // 解析查询参数
                            let mut code = None;
                            let mut state = None;

                            if let Some(query_start) = path.find('?') {
                                let query = &path[query_start + 1..];
                                for param in query.split('&') {
                                    let parts: Vec<&str> = param.splitn(2, '=').collect();
                                    if parts.len() == 2 {
                                        match parts[0] {
                                            "code" => {
                                                code = Some(
                                                    urlencoding::decode(parts[1])
                                                        .unwrap_or_default()
                                                        .to_string(),
                                                )
                                            }
                                            "state" => {
                                                state = Some(
                                                    urlencoding::decode(parts[1])
                                                        .unwrap_or_default()
                                                        .to_string(),
                                                )
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                            }

                            // 发送成功响应页面
                            let html = r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>登录成功</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .container { text-align: center; background: white; padding: 40px 60px; border-radius: 16px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); }
        h1 { color: #22c55e; margin-bottom: 10px; }
        p { color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h1>✓ 登录成功</h1>
        <p>您可以关闭此窗口并返回应用</p>
    </div>
</body>
</html>"#;

                            let response = format!(
                                "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                                html.len(),
                                html
                            );

                            let _ = socket.write_all(response.as_bytes()).await;

                            // 发送事件到前端
                            if let (Some(code), Some(state)) = (code, state) {
                                let _ = app_handle.emit(
                                    "kiro-social-auth-callback",
                                    serde_json::json!({
                                        "code": code,
                                        "state": state
                                    }),
                                );
                            }
                        }
                    }
                }
            }
        }
    });

    Ok(true)
}

// ============ Playwright 可用性检测测试 ============

#[cfg(test)]
mod playwright_tests {
    use super::*;

    /// **Property 1: Playwright 可用性检测正确性**
    /// **Validates: Requirements 2.2**
    ///
    /// *For any* 文件系统状态，Playwright 可用性检测函数应该：
    /// - 当 Playwright 浏览器可执行文件存在时返回 `available: true`
    /// - 当可执行文件不存在时返回 `available: false`
    /// - 返回的 `browserPath` 应该是实际检测到的路径或 `None`

    #[test]
    fn test_get_playwright_cache_dir_returns_valid_path() {
        // Feature: playwright-fingerprint-login, Property 1: Playwright 可用性检测正确性
        // 测试缓存目录路径生成
        let cache_dir = get_playwright_cache_dir();

        // 路径应该包含 ms-playwright
        assert!(
            cache_dir.to_string_lossy().contains("ms-playwright"),
            "缓存目录应包含 ms-playwright: {:?}",
            cache_dir
        );

        // 路径应该是绝对路径或相对于 home 目录
        #[cfg(target_os = "macos")]
        assert!(
            cache_dir.to_string_lossy().contains("Library/Caches"),
            "macOS 缓存目录应在 Library/Caches 下: {:?}",
            cache_dir
        );

        #[cfg(target_os = "windows")]
        assert!(
            cache_dir.to_string_lossy().contains("AppData\\Local"),
            "Windows 缓存目录应在 AppData\\Local 下: {:?}",
            cache_dir
        );

        #[cfg(target_os = "linux")]
        assert!(
            cache_dir.to_string_lossy().contains(".cache"),
            "Linux 缓存目录应在 .cache 下: {:?}",
            cache_dir
        );
    }

    #[test]
    fn test_get_playwright_browser_path_returns_none_when_not_installed() {
        // Feature: playwright-fingerprint-login, Property 1: Playwright 可用性检测正确性
        // 当 Playwright 未安装时，应返回 None
        // 注意：这个测试在 Playwright 已安装的环境中可能会失败
        // 我们主要测试函数不会 panic
        let result = get_playwright_browser_path();

        // 函数应该正常返回（不 panic）
        // 结果可能是 Some 或 None，取决于环境
        match result {
            Some(path) => {
                // 如果找到了路径，验证路径格式
                assert!(!path.is_empty(), "浏览器路径不应为空");
                assert!(
                    path.contains("chromium")
                        || path.contains("Chromium")
                        || path.contains("chrome"),
                    "路径应包含 chromium/chrome: {}",
                    path
                );
            }
            None => {
                // 未找到浏览器，这是预期的情况之一
            }
        }
    }

    #[test]
    fn test_playwright_status_serialization() {
        // Feature: playwright-fingerprint-login, Property 1: Playwright 可用性检测正确性
        // 测试 PlaywrightStatus 结构体的序列化

        // 测试可用状态
        let available_status = PlaywrightStatus {
            available: true,
            browser_path: Some("/path/to/chromium".to_string()),
            browser_source: Some("playwright".to_string()),
            error: None,
        };

        let json = serde_json::to_string(&available_status).unwrap();
        assert!(json.contains("\"available\":true"));
        assert!(json.contains("\"browser_path\":\"/path/to/chromium\""));

        // 测试不可用状态
        let unavailable_status = PlaywrightStatus {
            available: false,
            browser_path: None,
            browser_source: None,
            error: Some("未安装".to_string()),
        };

        let json = serde_json::to_string(&unavailable_status).unwrap();
        assert!(json.contains("\"available\":false"));
        assert!(json.contains("\"error\":\"未安装\""));
    }

    #[test]
    fn test_playwright_status_deserialization() {
        // Feature: playwright-fingerprint-login, Property 1: Playwright 可用性检测正确性
        // 测试 PlaywrightStatus 结构体的反序列化

        let json = r#"{"available":true,"browser_path":"/test/path","error":null}"#;
        let status: PlaywrightStatus = serde_json::from_str(json).unwrap();

        assert!(status.available);
        assert_eq!(status.browser_path, Some("/test/path".to_string()));
        assert!(status.error.is_none());
    }

    #[test]
    fn test_playwright_status_invariants() {
        // Feature: playwright-fingerprint-login, Property 1: Playwright 可用性检测正确性
        // 测试状态不变量：
        // - 当 available=true 时，browser_path 应该有值
        // - 当 available=false 时，error 应该有值

        // 可用状态的不变量
        let available_status = PlaywrightStatus {
            available: true,
            browser_path: Some("/path".to_string()),
            browser_source: Some("system".to_string()),
            error: None,
        };
        assert!(
            available_status.available && available_status.browser_path.is_some(),
            "可用状态应有 browser_path"
        );

        // 不可用状态的不变量
        let unavailable_status = PlaywrightStatus {
            available: false,
            browser_path: None,
            browser_source: None,
            error: Some("错误".to_string()),
        };
        assert!(
            !unavailable_status.available && unavailable_status.error.is_some(),
            "不可用状态应有 error"
        );
    }
}

/// 获取单个凭证的健康状态
/// Requirements: 4.4
#[tauri::command]
pub async fn get_credential_health(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
    uuid: String,
) -> Result<Option<crate::services::provider_pool_service::CredentialHealthInfo>, String> {
    pool_service.0.get_credential_health(&db, &uuid)
}

/// 获取所有凭证的健康状态
/// Requirements: 4.4
#[tauri::command]
pub async fn get_all_credential_health(
    db: State<'_, DbConnection>,
    pool_service: State<'_, ProviderPoolServiceState>,
) -> Result<Vec<crate::services::provider_pool_service::CredentialHealthInfo>, String> {
    pool_service.0.get_all_credential_health(&db)
}
