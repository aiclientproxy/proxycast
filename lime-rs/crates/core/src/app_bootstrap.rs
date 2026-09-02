//! 应用启动前的配置校验逻辑
//!
//! 该模块仅包含与桌面宿主无关的纯配置处理。

use crate::app_utils::{generate_api_key, is_valid_bind_host};
use crate::config::{self, Config};

/// 配置验证错误
#[derive(Debug)]
pub enum ConfigError {
    LoadFailed(String),
    SaveFailed(String),
    InvalidHost,
    DefaultApiKeyWithNonLocalBind,
    TlsNotSupported,
    RemoteManagementNotSupported,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::LoadFailed(e) => write!(f, "配置加载失败: {e}"),
            ConfigError::SaveFailed(e) => write!(f, "配置保存失败: {e}"),
            ConfigError::InvalidHost => {
                write!(
                    f,
                    "无效的监听地址。允许的地址：127.0.0.1、localhost、::1、0.0.0.0、::"
                )
            }
            ConfigError::DefaultApiKeyWithNonLocalBind => write!(
                f,
                "监听所有网络接口 (0.0.0.0 或 ::) 时，必须设置非默认的 API Key"
            ),
            ConfigError::TlsNotSupported => write!(f, "当前版本尚未支持 TLS"),
            ConfigError::RemoteManagementNotSupported => {
                write!(f, "远程管理需要 TLS 支持，当前版本未启用")
            }
        }
    }
}

impl std::error::Error for ConfigError {}

/// 加载并验证配置
pub fn load_and_validate_config() -> Result<Config, ConfigError> {
    let mut config = config::load_config().map_err(|e| ConfigError::LoadFailed(e.to_string()))?;

    if !is_valid_bind_host(&config.server.host) {
        return Err(ConfigError::InvalidHost);
    }

    if config.server.api_key == config::DEFAULT_API_KEY {
        let new_key = generate_api_key();
        config.server.api_key = new_key;
        config::save_config(&config).map_err(|e| ConfigError::SaveFailed(e.to_string()))?;
        tracing::info!("检测到默认 API key，已自动生成并保存新密钥");
    }

    if config.server.tls.enable {
        return Err(ConfigError::TlsNotSupported);
    }

    if config.remote_management.allow_remote {
        return Err(ConfigError::RemoteManagementNotSupported);
    }

    Ok(config)
}
