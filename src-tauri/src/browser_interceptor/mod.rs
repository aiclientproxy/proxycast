pub mod config;
pub mod interceptor;
pub mod notification_service;
pub mod state_manager;
pub mod url_manager;

#[cfg(target_os = "windows")]
pub mod platform {
    pub mod windows;
}

#[cfg(target_os = "macos")]
pub mod platform {
    pub mod macos;
}

#[cfg(target_os = "linux")]
pub mod platform {
    pub mod linux;
}

// 重新导出主要类型和函数
pub use config::{BrowserInterceptorConfig, InterceptedUrl, InterceptorState};
pub use interceptor::BrowserInterceptor;
pub use notification_service::NotificationService;
pub use state_manager::StateManager;
pub use url_manager::UrlManager;

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

/// 浏览器拦截器错误类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BrowserInterceptorError {
    ConfigError(String),
    InterceptorError(String),
    StateError(String),
    PlatformError(String),
    NotificationError(String),
    AlreadyRunning,
    UnsupportedPlatform(String),
    IoError(String),
}

impl fmt::Display for BrowserInterceptorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BrowserInterceptorError::ConfigError(msg) => write!(f, "配置错误: {}", msg),
            BrowserInterceptorError::InterceptorError(msg) => write!(f, "拦截器错误: {}", msg),
            BrowserInterceptorError::StateError(msg) => write!(f, "状态管理错误: {}", msg),
            BrowserInterceptorError::PlatformError(msg) => write!(f, "平台错误: {}", msg),
            BrowserInterceptorError::NotificationError(msg) => write!(f, "通知错误: {}", msg),
            BrowserInterceptorError::AlreadyRunning => write!(f, "拦截器已在运行"),
            BrowserInterceptorError::UnsupportedPlatform(msg) => write!(f, "不支持的平台: {}", msg),
            BrowserInterceptorError::IoError(msg) => write!(f, "IO错误: {}", msg),
        }
    }
}

impl Error for BrowserInterceptorError {}

impl From<std::io::Error> for BrowserInterceptorError {
    fn from(err: std::io::Error) -> Self {
        BrowserInterceptorError::IoError(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, BrowserInterceptorError>;
