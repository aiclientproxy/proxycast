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
