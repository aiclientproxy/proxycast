use crate::config::{
    Config, ConfigManager, ExportBundle, ExportOptions as ExportServiceOptions, ExportService,
    ImportOptions as ImportServiceOptions, ImportService, ValidationResult,
};
use crate::models::AppType;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::AppHandle;
use tauri_plugin_autostart::ManagerExt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigStatus {
    pub exists: bool,
    pub path: String,
    pub has_env: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionCheckResult {
    pub current: String,
    pub latest: Option<String>,
    #[serde(rename = "hasUpdate")]
    pub has_update: bool,
    #[serde(rename = "downloadUrl")]
    pub download_url: Option<String>,
    pub error: Option<String>,
}

/// Get the config directory path for an app type
fn get_config_dir(app_type: &AppType) -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    match app_type {
        AppType::Claude => Some(home.join(".claude")),
        AppType::Codex => Some(home.join(".codex")),
        AppType::Gemini => Some(home.join(".gemini")),
        AppType::ProxyCast => dirs::config_dir().map(|d| d.join("proxycast")),
    }
}

#[tauri::command]
pub fn get_config_status(app_type: String) -> Result<ConfigStatus, String> {
    let app = app_type.parse::<AppType>().map_err(|e| e.to_string())?;
    let config_dir = get_config_dir(&app).ok_or("Cannot determine config directory")?;

    let main_config = match app {
        AppType::Claude => config_dir.join("settings.json"),
        AppType::Codex => config_dir.join("auth.json"),
        AppType::Gemini => config_dir.join(".env"),
        AppType::ProxyCast => config_dir.join("config.json"),
    };

    let has_env = match app {
        AppType::Claude => {
            config_dir.join("settings.json").exists()
                && std::fs::read_to_string(config_dir.join("settings.json"))
                    .map(|s| s.contains("env"))
                    .unwrap_or(false)
        }
        AppType::Codex => config_dir.join("auth.json").exists(),
        AppType::Gemini => config_dir.join(".env").exists(),
        AppType::ProxyCast => config_dir.join("config.json").exists(),
    };

    Ok(ConfigStatus {
        exists: main_config.exists(),
        path: config_dir.to_string_lossy().to_string(),
        has_env,
    })
}

#[tauri::command]
pub fn get_config_dir_path(app_type: String) -> Result<String, String> {
    let app = app_type.parse::<AppType>().map_err(|e| e.to_string())?;
    let config_dir = get_config_dir(&app).ok_or("Cannot determine config directory")?;
    Ok(config_dir.to_string_lossy().to_string())
}

#[tauri::command]
pub async fn open_config_folder(_handle: AppHandle, app_type: String) -> Result<bool, String> {
    let app = app_type.parse::<AppType>().map_err(|e| e.to_string())?;
    let config_dir = get_config_dir(&app).ok_or("Cannot determine config directory")?;

    if !config_dir.exists() {
        std::fs::create_dir_all(&config_dir).map_err(|e| e.to_string())?;
    }

    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(&config_dir)
            .spawn()
            .map_err(|e| e.to_string())?;
    }

    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("explorer")
            .arg(&config_dir)
            .spawn()
            .map_err(|e| e.to_string())?;
    }

    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(&config_dir)
            .spawn()
            .map_err(|e| e.to_string())?;
    }

    Ok(true)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolVersion {
    pub name: String,
    pub version: Option<String>,
    pub installed: bool,
}

#[tauri::command]
pub async fn get_tool_versions() -> Result<Vec<ToolVersion>, String> {
    let mut versions = Vec::new();

    // Check Claude Code version
    let claude_version = std::process::Command::new("claude")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_string());

    versions.push(ToolVersion {
        name: "Claude Code".to_string(),
        version: claude_version.clone(),
        installed: claude_version.is_some(),
    });

    // Check Codex version
    let codex_version = std::process::Command::new("codex")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_string());

    versions.push(ToolVersion {
        name: "Codex".to_string(),
        version: codex_version.clone(),
        installed: codex_version.is_some(),
    });

    // Check Gemini CLI version
    let gemini_version = std::process::Command::new("gemini")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_string());

    versions.push(ToolVersion {
        name: "Gemini CLI".to_string(),
        version: gemini_version.clone(),
        installed: gemini_version.is_some(),
    });

    Ok(versions)
}

#[tauri::command]
pub async fn get_auto_launch_status(app: AppHandle) -> Result<bool, String> {
    let autostart_manager = app.autolaunch();
    autostart_manager
        .is_enabled()
        .map_err(|e| format!("Failed to get autostart status: {e}"))
}

#[tauri::command]
pub async fn set_auto_launch(app: AppHandle, enabled: bool) -> Result<bool, String> {
    let autostart_manager = app.autolaunch();

    if enabled {
        autostart_manager
            .enable()
            .map_err(|e| format!("Failed to enable autostart: {e}"))?;
    } else {
        autostart_manager
            .disable()
            .map_err(|e| format!("Failed to disable autostart: {e}"))?;
    }

    Ok(enabled)
}

<<<<<<< HEAD
// ============ Config Import/Export Commands ============

/// 配置导出选项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportOptions {
    /// 是否脱敏敏感信息（API 密钥等）
    pub redact_secrets: bool,
}

/// 配置导出结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportResult {
    /// YAML 配置内容
    pub content: String,
    /// 建议的文件名
    pub suggested_filename: String,
}

/// 导出配置为 YAML 字符串
///
/// # Arguments
/// * `config` - 当前配置
/// * `redact_secrets` - 是否脱敏敏感信息
#[tauri::command]
pub fn export_config(config: Config, redact_secrets: bool) -> Result<ExportResult, String> {
    let manager = ConfigManager::new(PathBuf::from("temp.yaml"));
    let mut manager_with_config = manager;
    manager_with_config.set_config(config);

    let content = manager_with_config
        .export(redact_secrets)
        .map_err(|e| e.to_string())?;

    // 生成带时间戳的文件名
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let suffix = if redact_secrets { "_redacted" } else { "" };
    let suggested_filename = format!("proxycast_config_{}{}.yaml", timestamp, suffix);

    Ok(ExportResult {
        content,
        suggested_filename,
    })
}

/// 配置导入选项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportOptions {
    /// 是否合并到现有配置（true）或替换（false）
    pub merge: bool,
}

/// 配置导入结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportResult {
    /// 是否成功
    pub success: bool,
    /// 导入后的配置
    pub config: Config,
    /// 警告信息（如果有）
    pub warnings: Vec<String>,
}

/// 验证配置 YAML 格式
///
/// # Arguments
/// * `yaml_content` - YAML 配置字符串
#[tauri::command]
pub fn validate_config_yaml(yaml_content: String) -> Result<Config, String> {
    ConfigManager::parse_yaml(&yaml_content).map_err(|e| e.to_string())
}

/// 导入配置
///
/// # Arguments
/// * `current_config` - 当前配置
/// * `yaml_content` - 要导入的 YAML 配置字符串
/// * `merge` - 是否合并到现有配置（true）或替换（false）
#[tauri::command]
pub fn import_config(
    current_config: Config,
    yaml_content: String,
    merge: bool,
) -> Result<ImportResult, String> {
    let mut manager = ConfigManager::new(PathBuf::from("temp.yaml"));
    manager.set_config(current_config);

    let mut warnings = Vec::new();

    // 先验证 YAML 格式
    let imported_config = ConfigManager::parse_yaml(&yaml_content).map_err(|e| e.to_string())?;

    // 检查是否包含脱敏的密钥
    if imported_config.server.api_key == "***REDACTED***" {
        warnings.push("导入的配置包含脱敏的 API 密钥，将保留原有值".to_string());
    }
    if imported_config
        .providers
        .openai
        .api_key
        .as_ref()
        .map(|k| k == "***REDACTED***")
        .unwrap_or(false)
    {
        warnings.push("导入的配置包含脱敏的 OpenAI API 密钥，将保留原有值".to_string());
    }
    if imported_config
        .providers
        .claude
        .api_key
        .as_ref()
        .map(|k| k == "***REDACTED***")
        .unwrap_or(false)
    {
        warnings.push("导入的配置包含脱敏的 Claude API 密钥，将保留原有值".to_string());
    }

    // 执行导入
    manager
        .import(&yaml_content, merge)
        .map_err(|e| e.to_string())?;

    // 如果导入的配置包含脱敏的密钥，恢复原有值
    let final_config = manager.config().clone();

    Ok(ImportResult {
        success: true,
        config: final_config,
        warnings,
    })
}

/// 获取配置文件路径信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigPathInfo {
    /// YAML 配置文件路径
    pub yaml_path: String,
    /// JSON 配置文件路径（旧版）
    pub json_path: String,
    /// YAML 配置是否存在
    pub yaml_exists: bool,
    /// JSON 配置是否存在
    pub json_exists: bool,
}

/// 获取配置文件路径信息
#[tauri::command]
pub fn get_config_paths() -> Result<ConfigPathInfo, String> {
    let yaml_path = ConfigManager::default_config_path();
    let json_path = dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("proxycast")
        .join("config.json");

    Ok(ConfigPathInfo {
        yaml_path: yaml_path.to_string_lossy().to_string(),
        json_path: json_path.to_string_lossy().to_string(),
        yaml_exists: yaml_path.exists(),
        json_exists: json_path.exists(),
    })
}

// ============ Enhanced Export/Import Commands (using ExportService/ImportService) ============

/// 统一导出选项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedExportOptions {
    /// 是否包含配置
    pub include_config: bool,
    /// 是否包含凭证
    pub include_credentials: bool,
    /// 是否脱敏敏感信息
    pub redact_secrets: bool,
}

/// 统一导出结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedExportResult {
    /// 导出包内容（JSON 格式）
    pub content: String,
    /// 建议的文件名
    pub suggested_filename: String,
    /// 是否已脱敏
    pub redacted: bool,
    /// 是否包含配置
    pub has_config: bool,
    /// 是否包含凭证
    pub has_credentials: bool,
}

/// 导出完整的配置和凭证包
///
/// # Arguments
/// * `config` - 当前配置
/// * `options` - 导出选项
///
/// # Requirements: 3.1, 3.2
#[tauri::command]
pub fn export_bundle(
    config: Config,
    options: UnifiedExportOptions,
) -> Result<UnifiedExportResult, String> {
    let export_options = ExportServiceOptions {
        include_config: options.include_config,
        include_credentials: options.include_credentials,
        redact_secrets: options.redact_secrets,
    };

    // 获取应用版本
    let app_version = env!("CARGO_PKG_VERSION").to_string();

    let bundle =
        ExportService::export(&config, &export_options, &app_version).map_err(|e| e.to_string())?;

    let content = bundle.to_json().map_err(|e| e.to_string())?;

    // 生成带时间戳的文件名
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let suffix = if options.redact_secrets {
        "_redacted"
    } else {
        ""
    };
    let scope = match (options.include_config, options.include_credentials) {
        (true, true) => "full",
        (true, false) => "config",
        (false, true) => "credentials",
        (false, false) => "empty",
    };
    let suggested_filename = format!("proxycast_{}_{}{}.json", scope, timestamp, suffix);

    Ok(UnifiedExportResult {
        content,
        suggested_filename,
        redacted: bundle.redacted,
        has_config: bundle.has_config(),
        has_credentials: bundle.has_credentials(),
    })
}

/// 仅导出配置为 YAML
///
/// # Arguments
/// * `config` - 当前配置
/// * `redact_secrets` - 是否脱敏敏感信息
///
/// # Requirements: 3.1, 5.1
#[tauri::command]
pub fn export_config_yaml(config: Config, redact_secrets: bool) -> Result<ExportResult, String> {
    let content = ExportService::export_yaml(&config, redact_secrets).map_err(|e| e.to_string())?;

    // 生成带时间戳的文件名
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let suffix = if redact_secrets { "_redacted" } else { "" };
    let suggested_filename = format!("proxycast_config_{}{}.yaml", timestamp, suffix);

    Ok(ExportResult {
        content,
        suggested_filename,
    })
}

/// 验证导入内容
///
/// # Arguments
/// * `content` - 导入内容（JSON 导出包或 YAML 配置）
///
/// # Requirements: 4.1, 4.2
#[tauri::command]
pub fn validate_import(content: String) -> Result<ValidationResult, String> {
    Ok(ImportService::validate(&content))
}

/// 导入完整的导出包
///
/// # Arguments
/// * `current_config` - 当前配置
/// * `content` - 导出包内容（JSON 格式）
/// * `merge` - 是否合并到现有配置
///
/// # Requirements: 4.1, 4.3
#[tauri::command]
pub fn import_bundle(
    current_config: Config,
    content: String,
    merge: bool,
) -> Result<ImportResult, String> {
    // 首先尝试解析为 ExportBundle
    if let Ok(bundle) = ExportBundle::from_json(&content) {
        let options = ImportServiceOptions { merge };
        let result =
            ImportService::import(&bundle, &current_config, &options, &current_config.auth_dir)
                .map_err(|e| e.to_string())?;

        return Ok(ImportResult {
            success: result.success,
            config: result.config,
            warnings: result.warnings,
        });
    }

    // 尝试解析为 YAML 配置
    let options = ImportServiceOptions { merge };
    let result = ImportService::import_yaml(&content, &current_config, &options)
        .map_err(|e| e.to_string())?;

    Ok(ImportResult {
        success: result.success,
        config: result.config,
        warnings: result.warnings,
    })
}

// ============ Path Utility Commands ============

/// 展开路径中的 tilde (~) 为用户主目录
///
/// # Arguments
/// * `path` - 要展开的路径字符串
///
/// # Returns
/// 展开后的完整路径字符串
///
/// # Requirements: 2.3
#[tauri::command]
pub fn expand_path(path: String) -> Result<String, String> {
    use crate::config::expand_tilde;

    let expanded = expand_tilde(&path);
    Ok(expanded.to_string_lossy().to_string())
}

/// 打开认证目录
///
/// # Arguments
/// * `path` - 认证目录路径（支持 tilde 展开）
///
/// # Requirements: 2.2
#[tauri::command]
pub async fn open_auth_dir(path: String) -> Result<bool, String> {
    use crate::config::expand_tilde;

    let expanded = expand_tilde(&path);

    // 确保目录存在
    if !expanded.exists() {
        std::fs::create_dir_all(&expanded).map_err(|e| e.to_string())?;
    }

    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(&expanded)
            .spawn()
            .map_err(|e| e.to_string())?;
    }

    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("explorer")
            .arg(&expanded)
            .spawn()
            .map_err(|e| e.to_string())?;
    }

    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(&expanded)
            .spawn()
            .map_err(|e| e.to_string())?;
    }

    Ok(true)
}

#[tauri::command]
pub async fn check_for_updates() -> Result<VersionCheckResult, String> {
    const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");
    const GITHUB_API_URL: &str =
        "https://api.github.com/repos/aiclientproxy/proxycast/releases/latest";

    let client = reqwest::Client::new();

    match client
        .get(GITHUB_API_URL)
        .header("User-Agent", "ProxyCast")
        .send()
        .await
    {
        Ok(response) => {
            if response.status().is_success() {
                match response.json::<serde_json::Value>().await {
                    Ok(data) => {
                        let latest_version = data["tag_name"]
                            .as_str()
                            .unwrap_or("")
                            .trim_start_matches('v');

                        let download_url = data["html_url"].as_str().map(|s| s.to_string());

                        let has_update = version_compare(CURRENT_VERSION, latest_version);

                        Ok(VersionCheckResult {
                            current: CURRENT_VERSION.to_string(),
                            latest: Some(latest_version.to_string()),
                            has_update,
                            download_url,
                            error: None,
                        })
                    }
                    Err(e) => Ok(VersionCheckResult {
                        current: CURRENT_VERSION.to_string(),
                        latest: None,
                        has_update: false,
                        download_url: None,
                        error: Some(format!("解析响应失败: {}", e)),
                    }),
                }
            } else {
                Ok(VersionCheckResult {
                    current: CURRENT_VERSION.to_string(),
                    latest: None,
                    has_update: false,
                    download_url: None,
                    error: Some(format!("GitHub API 请求失败: {}", response.status())),
                })
            }
        }
        Err(e) => Ok(VersionCheckResult {
            current: CURRENT_VERSION.to_string(),
            latest: None,
            has_update: false,
            download_url: None,
            error: Some(format!("网络请求失败: {}", e)),
        }),
    }
}

/// 简单的版本比较函数
/// 返回 true 如果 latest > current
fn version_compare(current: &str, latest: &str) -> bool {
    // 移除 'v' 前缀
    let current = current.trim_start_matches('v');
    let latest = latest.trim_start_matches('v');

    let current_parts: Vec<u32> = current.split('.').filter_map(|s| s.parse().ok()).collect();
    let latest_parts: Vec<u32> = latest.split('.').filter_map(|s| s.parse().ok()).collect();

    let max_len = current_parts.len().max(latest_parts.len());

    for i in 0..max_len {
        let current_part = current_parts.get(i).unwrap_or(&0);
        let latest_part = latest_parts.get(i).unwrap_or(&0);

        if latest_part > current_part {
            return true;
        } else if latest_part < current_part {
            return false;
        }
    }

    false
}
