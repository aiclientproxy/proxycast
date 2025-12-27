use crate::browser_interceptor::{BrowserInterceptorError, InterceptorState, Result};
use chrono::Utc;
use std::sync::{Arc, RwLock};
use tokio::time::{Duration, Instant};

#[cfg(windows)]
use {
    std::ffi::OsStr,
    std::iter::once,
    std::os::windows::ffi::OsStrExt,
    windows::{core::*, Win32::Foundation::*, Win32::System::Registry::*},
};

/// 状态管理器，负责管理拦截器的状态和恢复机制
pub struct StateManager {
    state: Arc<RwLock<InterceptorState>>,
    original_system_state: Arc<RwLock<Option<SystemState>>>,
    temporary_disable_timer: Arc<RwLock<Option<Instant>>>,
}

/// 系统原始状态备份
#[derive(Debug, Clone)]
pub struct SystemState {
    pub default_browser: Option<String>,
    pub registry_backup: std::collections::HashMap<String, String>,
    pub environment_backup: std::collections::HashMap<String, String>,
    pub timestamp: chrono::DateTime<Utc>,
}

impl StateManager {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(InterceptorState::default())),
            original_system_state: Arc::new(RwLock::new(None)),
            temporary_disable_timer: Arc::new(RwLock::new(None)),
        }
    }

    /// 获取当前状态
    pub fn get_state(&self) -> Result<InterceptorState> {
        self.state
            .read()
            .map_err(|e| BrowserInterceptorError::StateError(format!("读取状态失败: {}", e)))
            .map(|state| state.clone())
    }

    /// 启用拦截器
    pub async fn enable_interceptor(&self) -> Result<()> {
        // 备份系统状态
        self.backup_system_state().await?;

        // 更新状态
        {
            let mut state = self
                .state
                .write()
                .map_err(|e| BrowserInterceptorError::StateError(format!("写入状态失败: {}", e)))?;

            state.enabled = true;
            state.can_restore = true;
            state.last_activity = Some(Utc::now());
        }

        tracing::info!("浏览器拦截器已启用");
        Ok(())
    }

    /// 禁用拦截器
    pub async fn disable_interceptor(&self) -> Result<()> {
        {
            let mut state = self
                .state
                .write()
                .map_err(|e| BrowserInterceptorError::StateError(format!("写入状态失败: {}", e)))?;

            state.enabled = false;
            state.active_hooks.clear();
            state.last_activity = Some(Utc::now());
        }

        tracing::info!("浏览器拦截器已禁用");
        Ok(())
    }

    /// 临时禁用拦截器
    pub async fn temporary_disable(&self, duration_seconds: u64) -> Result<()> {
        self.disable_interceptor().await?;

        // 设置定时器
        {
            let mut timer = self.temporary_disable_timer.write().map_err(|e| {
                BrowserInterceptorError::StateError(format!("设置定时器失败: {}", e))
            })?;
            *timer = Some(Instant::now() + Duration::from_secs(duration_seconds));
        }

        // 启动后台任务来重新启用
        let state_manager = self.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(duration_seconds)).await;
            if let Err(e) = state_manager.enable_interceptor().await {
                tracing::error!("自动重新启用拦截器失败: {}", e);
            } else {
                tracing::info!("拦截器已自动重新启用");
            }
        });

        tracing::info!("拦截器已临时禁用 {} 秒", duration_seconds);
        Ok(())
    }

    /// 恢复正常浏览器行为
    pub async fn restore_normal_behavior(&self) -> Result<()> {
        // 先禁用拦截器
        self.disable_interceptor().await?;

        // 恢复系统状态
        self.restore_system_state().await?;

        {
            let mut state = self
                .state
                .write()
                .map_err(|e| BrowserInterceptorError::StateError(format!("写入状态失败: {}", e)))?;
            state.can_restore = false;
        }

        tracing::info!("已恢复正常浏览器行为");
        Ok(())
    }

    /// 增加拦截计数
    pub fn increment_intercept_count(&self) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|e| BrowserInterceptorError::StateError(format!("写入状态失败: {}", e)))?;

        state.intercepted_count += 1;
        state.last_activity = Some(Utc::now());

        Ok(())
    }

    /// 添加活跃钩子
    pub fn add_active_hook(&self, hook_name: String) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|e| BrowserInterceptorError::StateError(format!("写入状态失败: {}", e)))?;

        if !state.active_hooks.contains(&hook_name) {
            state.active_hooks.push(hook_name);
        }

        Ok(())
    }

    /// 移除活跃钩子
    pub fn remove_active_hook(&self, hook_name: &str) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|e| BrowserInterceptorError::StateError(format!("写入状态失败: {}", e)))?;

        state.active_hooks.retain(|h| h != hook_name);

        Ok(())
    }

    /// 备份系统状态
    async fn backup_system_state(&self) -> Result<()> {
        let system_state = SystemState {
            default_browser: self.get_default_browser().await?,
            registry_backup: self.backup_registry_keys().await?,
            environment_backup: self.backup_environment_variables().await?,
            timestamp: Utc::now(),
        };

        {
            let mut backup = self.original_system_state.write().map_err(|e| {
                BrowserInterceptorError::StateError(format!("备份系统状态失败: {}", e))
            })?;
            *backup = Some(system_state);
        }

        tracing::info!("系统状态已备份");
        Ok(())
    }

    /// 恢复系统状态
    async fn restore_system_state(&self) -> Result<()> {
        let backup = {
            let backup_guard = self.original_system_state.read().map_err(|e| {
                BrowserInterceptorError::StateError(format!("读取备份状态失败: {}", e))
            })?;
            backup_guard.clone()
        };

        if let Some(system_state) = backup {
            // 恢复默认浏览器
            if let Some(default_browser) = &system_state.default_browser {
                self.restore_default_browser(default_browser).await?;
            }

            // 恢复注册表项
            self.restore_registry_keys(&system_state.registry_backup)
                .await?;

            // 恢复环境变量
            self.restore_environment_variables(&system_state.environment_backup)
                .await?;

            tracing::info!("系统状态已恢复到 {} 的备份", system_state.timestamp);
        } else {
            tracing::warn!("没有找到系统状态备份");
        }

        Ok(())
    }

    /// 获取默认浏览器（平台特定实现）
    async fn get_default_browser(&self) -> Result<Option<String>> {
        #[cfg(target_os = "windows")]
        {
            self.get_default_browser_windows().await
        }

        #[cfg(target_os = "macos")]
        {
            // macOS 实现
            // TODO: 使用 Launch Services API
            Ok(None)
        }

        #[cfg(target_os = "linux")]
        {
            // Linux 实现
            // TODO: 读取 xdg-settings
            Ok(None)
        }

        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        {
            Ok(None)
        }
    }

    /// Windows 平台获取默认浏览器
    #[cfg(target_os = "windows")]
    async fn get_default_browser_windows(&self) -> Result<Option<String>> {
        unsafe {
            let key_path = to_wide_chars(
                r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\http\UserChoice",
            );
            let mut key: HKEY = HKEY::default();

            let result = RegOpenKeyExW(
                HKEY_CURRENT_USER,
                PCWSTR::from_raw(key_path.as_ptr()),
                0,
                KEY_READ,
                &mut key,
            );

            if result.is_ok() {
                let mut buffer = vec![0u16; 1024];
                let mut buffer_size = buffer.len() * 2;

                let mut buffer_size_u32 = buffer_size as u32;
                let result = RegQueryValueExW(
                    key,
                    PCWSTR::from_raw(to_wide_chars("ProgId").as_ptr()),
                    None,
                    None,
                    Some(buffer.as_mut_ptr() as *mut u8),
                    Some(&mut buffer_size_u32),
                );

                RegCloseKey(key);

                if result.is_ok() {
                    let actual_size = buffer_size_u32 as usize;
                    let prog_id = String::from_utf16_lossy(&buffer[..actual_size / 2 - 1]);
                    return Ok(Some(prog_id));
                }
            }
        }

        Ok(None)
    }

    /// 备份注册表项（Windows 特定）
    async fn backup_registry_keys(&self) -> Result<std::collections::HashMap<String, String>> {
        let mut backup = std::collections::HashMap::new();

        #[cfg(target_os = "windows")]
        {
            // 备份 HTTP 和 HTTPS 协议关联
            let keys_to_backup = [
                r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\http\UserChoice",
                r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\https\UserChoice",
            ];

            for key_path in &keys_to_backup {
                if let Ok(value) = self.read_registry_value(key_path, "ProgId").await {
                    backup.insert(key_path.to_string(), value);
                }
            }
        }

        Ok(backup)
    }

    /// 读取注册表值
    #[cfg(target_os = "windows")]
    async fn read_registry_value(&self, key_path: &str, value_name: &str) -> Result<String> {
        unsafe {
            let key_path_wide = to_wide_chars(key_path);
            let mut key: HKEY = HKEY::default();

            let result = RegOpenKeyExW(
                HKEY_CURRENT_USER,
                PCWSTR::from_raw(key_path_wide.as_ptr()),
                0,
                KEY_READ,
                &mut key,
            );

            if result.is_ok() {
                let mut buffer = vec![0u16; 1024];
                let mut buffer_size = buffer.len() * 2;

                let mut buffer_size_u32 = buffer_size as u32;
                let result = RegQueryValueExW(
                    key,
                    PCWSTR::from_raw(to_wide_chars(value_name).as_ptr()),
                    None,
                    None,
                    Some(buffer.as_mut_ptr() as *mut u8),
                    Some(&mut buffer_size_u32),
                );

                RegCloseKey(key);

                if result.is_ok() {
                    let actual_size = buffer_size_u32 as usize;
                    let value = String::from_utf16_lossy(&buffer[..actual_size / 2 - 1]);
                    return Ok(value);
                }
            }
        }

        Err(BrowserInterceptorError::StateError(format!(
            "无法读取注册表项: {}",
            key_path
        )))
    }

    /// 备份环境变量
    async fn backup_environment_variables(
        &self,
    ) -> Result<std::collections::HashMap<String, String>> {
        let mut backup = std::collections::HashMap::new();

        // 备份可能影响浏览器启动的环境变量
        if let Ok(browser) = std::env::var("BROWSER") {
            backup.insert("BROWSER".to_string(), browser);
        }

        Ok(backup)
    }

    /// 恢复默认浏览器
    async fn restore_default_browser(&self, browser: &str) -> Result<()> {
        #[cfg(target_os = "windows")]
        {
            self.restore_default_browser_windows(browser).await
        }

        #[cfg(not(target_os = "windows"))]
        {
            tracing::info!("恢复默认浏览器: {} (非 Windows 平台，跳过)", browser);
            Ok(())
        }
    }

    /// Windows 平台恢复默认浏览器
    #[cfg(target_os = "windows")]
    async fn restore_default_browser_windows(&self, browser: &str) -> Result<()> {
        // 恢复 HTTP 协议处理器
        self.write_registry_value(
            r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\http\UserChoice",
            "ProgId",
            browser,
        )
        .await?;

        // 恢复 HTTPS 协议处理器
        self.write_registry_value(
            r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\https\UserChoice",
            "ProgId",
            browser,
        )
        .await?;

        tracing::info!("已恢复默认浏览器为: {}", browser);
        Ok(())
    }

    /// 恢复注册表项
    async fn restore_registry_keys(
        &self,
        backup: &std::collections::HashMap<String, String>,
    ) -> Result<()> {
        #[cfg(target_os = "windows")]
        {
            for (key_path, value) in backup {
                if let Err(e) = self.write_registry_value(key_path, "ProgId", value).await {
                    tracing::error!("恢复注册表项失败 {}: {}", key_path, e);
                }
            }
        }

        Ok(())
    }

    /// 写入注册表值
    #[cfg(target_os = "windows")]
    async fn write_registry_value(
        &self,
        key_path: &str,
        value_name: &str,
        value_data: &str,
    ) -> Result<()> {
        unsafe {
            let key_path_wide = to_wide_chars(key_path);
            let mut key: HKEY = HKEY::default();

            let result = RegCreateKeyExW(
                HKEY_CURRENT_USER,
                PCWSTR::from_raw(key_path_wide.as_ptr()),
                0,
                PCWSTR::null(),
                REG_OPTION_NON_VOLATILE,
                KEY_WRITE,
                None,
                &mut key,
                None,
            );

            if result.is_ok() {
                let value_name_wide = to_wide_chars(value_name);
                let value_data_wide = to_wide_chars(value_data);

                let data_bytes = std::slice::from_raw_parts(
                    value_data_wide.as_ptr() as *const u8,
                    value_data_wide.len() * 2,
                );

                RegSetValueExW(
                    key,
                    PCWSTR::from_raw(value_name_wide.as_ptr()),
                    0,
                    REG_SZ,
                    Some(data_bytes),
                );

                RegCloseKey(key);
                return Ok(());
            }
        }

        Err(BrowserInterceptorError::StateError(format!(
            "无法写入注册表项: {}",
            key_path
        )))
    }

    /// 恢复环境变量
    async fn restore_environment_variables(
        &self,
        backup: &std::collections::HashMap<String, String>,
    ) -> Result<()> {
        for (key, value) in backup {
            std::env::set_var(key, value);
        }
        Ok(())
    }
}

impl Clone for StateManager {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
            original_system_state: Arc::clone(&self.original_system_state),
            temporary_disable_timer: Arc::clone(&self.temporary_disable_timer),
        }
    }
}

impl Default for StateManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Windows 辅助函数：将字符串转换为宽字符
#[cfg(target_os = "windows")]
fn to_wide_chars(s: &str) -> Vec<u16> {
    OsStr::new(s).encode_wide().chain(once(0)).collect()
}
