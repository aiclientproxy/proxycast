use crate::browser_interceptor::{BrowserInterceptorError, InterceptedUrl, Result};
use chrono::Utc;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use uuid::Uuid;

#[cfg(windows)]
use {
    std::ffi::{OsStr, OsString},
    std::iter::once,
    std::os::windows::ffi::OsStrExt,
    std::ptr,
    windows::{
        core::*, Win32::Foundation::*, Win32::System::Registry::*, Win32::System::Threading::*,
        Win32::UI::Shell::*,
    },
};

/// Windows 平台的浏览器拦截器
pub struct WindowsInterceptor {
    running: bool,
    original_browser: Option<String>,
    temp_exe_path: Option<String>,
    intercepted_urls_handler: Arc<Mutex<Box<dyn Fn(InterceptedUrl) + Send + Sync>>>,
    monitor_thread: Option<std::thread::JoinHandle<()>>,
}

#[cfg(windows)]
impl WindowsInterceptor {
    pub fn new<F>(url_handler: F) -> Self
    where
        F: Fn(InterceptedUrl) + Send + Sync + 'static,
    {
        Self {
            running: false,
            original_browser: None,
            temp_exe_path: None,
            intercepted_urls_handler: Arc::new(Mutex::new(Box::new(url_handler))),
            monitor_thread: None,
        }
    }

    /// 启动拦截
    pub async fn start(&mut self) -> Result<()> {
        if self.running {
            return Err(BrowserInterceptorError::AlreadyRunning);
        }

        // 备份当前默认浏览器设置
        self.backup_default_browser().await?;

        // 创建临时拦截程序
        self.create_interceptor_executable().await?;

        // 设置我们的程序为默认浏览器
        self.set_as_default_browser().await?;

        // 启动进程监控
        self.start_process_monitoring().await?;

        self.running = true;
        tracing::info!("Windows 浏览器拦截器已启动");
        Ok(())
    }

    /// 停止拦截
    pub async fn stop(&mut self) -> Result<()> {
        if !self.running {
            return Ok(());
        }

        // 停止进程监控
        if let Some(handle) = self.monitor_thread.take() {
            // 发送停止信号，等待线程结束
            handle.join().ok();
        }

        // 恢复原始默认浏览器
        self.restore_default_browser().await?;

        // 清理临时文件
        self.cleanup_temp_files().await?;

        self.running = false;
        tracing::info!("Windows 浏览器拦截器已停止");
        Ok(())
    }

    /// 备份当前默认浏览器设置
    async fn backup_default_browser(&mut self) -> Result<()> {
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
                    self.original_browser = Some(prog_id);
                    tracing::info!(
                        "备份原始默认浏览器: {}",
                        self.original_browser.as_ref().unwrap()
                    );
                }
            }
        }

        Ok(())
    }

    /// 创建临时的拦截器可执行文件
    async fn create_interceptor_executable(&mut self) -> Result<()> {
        // 创建一个简单的拦截器程序，用于接收 URL 参数
        let temp_dir = std::env::temp_dir();
        let exe_path = temp_dir.join("proxycast_browser_interceptor.exe");

        // 创建拦截器脚本内容（批处理脚本）
        let bat_content = format!(
            r#"@echo off
echo URL被拦截: %1 >> "{}\proxycast_intercepted_urls.log"
"#,
            temp_dir.to_string_lossy()
        );

        let bat_path = temp_dir.join("proxycast_browser_interceptor.bat");
        std::fs::write(&bat_path, bat_content)?;

        // 创建一个简单的可执行文件包装器
        // 这里我们使用批处理文件，在真实环境中应该编译一个专用的小程序
        self.temp_exe_path = Some(bat_path.to_string_lossy().to_string());

        Ok(())
    }

    /// 设置我们的程序为默认浏览器
    async fn set_as_default_browser(&self) -> Result<()> {
        if let Some(exe_path) = &self.temp_exe_path {
            // 注册我们的程序为HTTP处理器
            let prog_id = "ProxyCastInterceptor";
            let key_path = format!(r"Software\Classes\{}", prog_id);

            self.set_registry_string(&key_path, "", "ProxyCast Browser Interceptor")
                .await?;

            let command_path = format!(r"{}\shell\open\command", key_path);
            let command_value = format!(r#"{} "%1""#, exe_path);
            self.set_registry_string(&command_path, "", &command_value)
                .await?;

            // 设置为HTTP协议的默认处理器
            let http_key =
                r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\http\UserChoice";
            self.set_registry_string(http_key, "ProgId", prog_id)
                .await?;

            // 设置为HTTPS协议的默认处理器
            let https_key =
                r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\https\UserChoice";
            self.set_registry_string(https_key, "ProgId", prog_id)
                .await?;

            tracing::info!("已设置 ProxyCast 为临时默认浏览器");
        }

        Ok(())
    }

    /// 设置注册表字符串值
    async fn set_registry_string(
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
            }
        }

        Ok(())
    }

    /// 启动进程监控
    async fn start_process_monitoring(&mut self) -> Result<()> {
        let handler = Arc::clone(&self.intercepted_urls_handler);
        let temp_dir = std::env::temp_dir();
        let log_file = temp_dir.join("proxycast_intercepted_urls.log");

        let handle = thread::spawn(move || {
            loop {
                // 检查拦截日志文件
                if let Ok(content) = std::fs::read_to_string(&log_file) {
                    for line in content.lines() {
                        if line.starts_with("URL被拦截: ") {
                            let url = line.replace("URL被拦截: ", "");
                            if !url.is_empty() && should_intercept_url(&url) {
                                let intercepted = InterceptedUrl {
                                    id: Uuid::new_v4().to_string(),
                                    url: url.clone(),
                                    source_process: "Unknown".to_string(),
                                    timestamp: Utc::now(),
                                    copied: false,
                                    opened_in_browser: false,
                                    dismissed: false,
                                };

                                // 调用处理器
                                if let Ok(handler_guard) = handler.lock() {
                                    handler_guard(intercepted);
                                }
                            }
                        }
                    }

                    // 清空日志文件避免重复处理
                    std::fs::write(&log_file, "").ok();
                }

                thread::sleep(Duration::from_millis(1000));
            }
        });

        self.monitor_thread = Some(handle);
        Ok(())
    }

    /// 恢复原始默认浏览器
    async fn restore_default_browser(&self) -> Result<()> {
        if let Some(original_browser) = &self.original_browser {
            // 恢复HTTP协议处理器
            let http_key =
                r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\http\UserChoice";
            self.set_registry_string(http_key, "ProgId", original_browser)
                .await?;

            // 恢复HTTPS协议处理器
            let https_key =
                r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\https\UserChoice";
            self.set_registry_string(https_key, "ProgId", original_browser)
                .await?;

            tracing::info!("已恢复原始默认浏览器: {}", original_browser);
        }

        Ok(())
    }

    /// 清理临时文件
    async fn cleanup_temp_files(&self) -> Result<()> {
        if let Some(exe_path) = &self.temp_exe_path {
            std::fs::remove_file(exe_path).ok();
        }

        let temp_dir = std::env::temp_dir();
        let log_file = temp_dir.join("proxycast_intercepted_urls.log");
        std::fs::remove_file(log_file).ok();

        Ok(())
    }

    /// 检查是否正在拦截
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// 恢复系统默认设置
    pub async fn restore_system_defaults(&self) -> Result<()> {
        // 恢复默认浏览器设置
        self.restore_default_browser().await?;
        tracing::info!("系统默认设置已恢复");
        Ok(())
    }

    /// 临时禁用拦截
    pub async fn temporarily_disable(&mut self) -> Result<()> {
        if let Some(_original_browser) = &self.original_browser {
            self.restore_default_browser().await?;
            tracing::info!("拦截器已临时禁用");
        }
        Ok(())
    }

    /// 重新启用拦截
    pub async fn re_enable(&mut self) -> Result<()> {
        self.set_as_default_browser().await?;
        tracing::info!("拦截器已重新启用");
        Ok(())
    }
}

#[cfg(not(windows))]
impl WindowsInterceptor {
    pub fn new<F>(_url_handler: F) -> Self
    where
        F: Fn(InterceptedUrl) + Send + Sync + 'static,
    {
        Self {
            running: false,
            original_browser: None,
            temp_exe_path: None,
            intercepted_urls_handler: Arc::new(Mutex::new(Box::new(|_| {}))),
            monitor_thread: None,
        }
    }

    pub async fn start(&mut self) -> Result<()> {
        Err(BrowserInterceptorError::UnsupportedPlatform(
            "Windows interceptor only supports Windows platform".to_string(),
        ))
    }

    pub async fn stop(&mut self) -> Result<()> {
        Ok(())
    }

    pub fn is_running(&self) -> bool {
        false
    }

    pub async fn restore_system_defaults(&self) -> Result<()> {
        Ok(())
    }

    pub async fn temporarily_disable(&mut self) -> Result<()> {
        Ok(())
    }

    pub async fn re_enable(&mut self) -> Result<()> {
        Ok(())
    }
}

impl Drop for WindowsInterceptor {
    fn drop(&mut self) {
        if self.running {
            // 在析构时恢复系统默认设置
            tokio::runtime::Handle::try_current().map(|handle| {
                handle.block_on(async {
                    let _ = self.stop().await;
                    let _ = self.restore_system_defaults().await;
                })
            });
        }
    }
}

// Windows 特定的辅助函数
fn to_wide_chars(s: &str) -> Vec<u16> {
    OsStr::new(s).encode_wide().chain(once(0)).collect()
}

/// 检查进程是否为目标应用
pub fn is_target_process(process_name: &str) -> bool {
    let target_processes = [
        "kiro",
        "kiro.exe",
        "cursor",
        "cursor.exe",
        "code",
        "code.exe",
    ];
    target_processes
        .iter()
        .any(|&target| process_name.to_lowercase().contains(&target.to_lowercase()))
}

/// 检查 URL 是否匹配拦截模式
pub fn should_intercept_url(url: &str) -> bool {
    let patterns = [
        "https://auth.",
        "https://accounts.google.com",
        "https://github.com/login",
        "https://login.microsoftonline.com",
        "/oauth/",
        "/auth/",
        "localhost:8080/auth", // OAuth 回调地址
    ];

    patterns.iter().any(|&pattern| url.contains(pattern))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_target_process() {
        assert!(is_target_process("kiro.exe"));
        assert!(is_target_process("cursor"));
        assert!(is_target_process("code.exe"));
        assert!(!is_target_process("notepad.exe"));
    }

    #[test]
    fn test_should_intercept_url() {
        assert!(should_intercept_url("https://accounts.google.com/oauth"));
        assert!(should_intercept_url("https://github.com/login/oauth"));
        assert!(should_intercept_url("localhost:8080/auth/callback"));
        assert!(!should_intercept_url("https://example.com"));
    }
}
