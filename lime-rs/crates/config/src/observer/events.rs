//! 配置变更事件类型
//!
//! 定义所有配置变更事件，支持细粒度的变更通知

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 配置变更事件类型
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum ConfigChangeEvent {
    /// 完整配置重载
    FullReload(FullReloadEvent),
    /// 路由配置变更
    RoutingChanged(RoutingChangeEvent),
    /// 注入配置变更
    InjectionChanged(InjectionChangeEvent),
    /// 端点 Provider 配置变更
    EndpointProvidersChanged(EndpointProvidersChangeEvent),
    /// 服务器配置变更
    ServerChanged(ServerChangeEvent),
    /// 日志配置变更
    LoggingChanged(LoggingChangeEvent),
    /// 重试配置变更
    RetryChanged(RetryChangeEvent),
    /// Amp CLI 配置变更
    AmpConfigChanged(AmpConfigChangeEvent),
    /// Native Agent 配置变更
    NativeAgentChanged(NativeAgentChangeEvent),
}

impl ConfigChangeEvent {
    /// 获取事件类型名称
    pub fn event_type(&self) -> &'static str {
        match self {
            ConfigChangeEvent::FullReload(_) => "FullReload",
            ConfigChangeEvent::RoutingChanged(_) => "RoutingChanged",
            ConfigChangeEvent::InjectionChanged(_) => "InjectionChanged",
            ConfigChangeEvent::EndpointProvidersChanged(_) => "EndpointProvidersChanged",
            ConfigChangeEvent::ServerChanged(_) => "ServerChanged",
            ConfigChangeEvent::LoggingChanged(_) => "LoggingChanged",
            ConfigChangeEvent::RetryChanged(_) => "RetryChanged",
            ConfigChangeEvent::AmpConfigChanged(_) => "AmpConfigChanged",
            ConfigChangeEvent::NativeAgentChanged(_) => "NativeAgentChanged",
        }
    }
}

/// 配置变更来源
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConfigChangeSource {
    /// 文件热重载
    HotReload,
    /// App Server / API 调用
    ApiCall,
    /// 前端 UI
    FrontendUI,
    /// 系统初始化
    SystemInit,
}

impl std::fmt::Display for ConfigChangeSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigChangeSource::HotReload => write!(f, "HotReload"),
            ConfigChangeSource::ApiCall => write!(f, "ApiCall"),
            ConfigChangeSource::FrontendUI => write!(f, "FrontendUI"),
            ConfigChangeSource::SystemInit => write!(f, "SystemInit"),
        }
    }
}

/// 完整配置重载事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullReloadEvent {
    /// 重载时间戳（毫秒）
    pub timestamp_ms: u64,
    /// 变更来源
    pub source: ConfigChangeSource,
}

/// 路由配置变更事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingChangeEvent {
    /// 默认 Provider（如果变更）
    pub default_provider: Option<String>,
    /// 模型别名是否变更
    pub model_aliases_changed: bool,
    /// 新的模型别名映射（如果变更）
    pub model_aliases: Option<HashMap<String, String>>,
    /// 变更来源
    pub source: ConfigChangeSource,
}

/// 注入配置变更事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionChangeEvent {
    /// 是否启用
    pub enabled: bool,
    /// 规则数量
    pub rules_count: usize,
    /// 变更来源
    pub source: ConfigChangeSource,
}

/// 端点 Provider 配置变更事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointProvidersChangeEvent {
    pub cursor: Option<String>,
    pub claude_code: Option<String>,
    pub codex: Option<String>,
    pub windsurf: Option<String>,
    pub kiro: Option<String>,
    pub other: Option<String>,
    /// 变更来源
    pub source: ConfigChangeSource,
}

/// 服务器配置变更事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerChangeEvent {
    /// API Key 是否变更
    pub api_key_changed: bool,
    /// Host 是否变更
    pub host_changed: bool,
    /// Port 是否变更
    pub port_changed: bool,
    /// 新的 Host（如果变更）
    pub new_host: Option<String>,
    /// 新的 Port（如果变更）
    pub new_port: Option<u16>,
    /// 变更来源
    pub source: ConfigChangeSource,
}

/// 日志配置变更事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingChangeEvent {
    pub enabled: bool,
    pub level: String,
    pub retention_days: u32,
    /// 变更来源
    pub source: ConfigChangeSource,
}

/// 重试配置变更事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryChangeEvent {
    pub max_retries: u32,
    pub base_delay_ms: u64,
    pub max_delay_ms: u64,
    pub auto_switch_provider: bool,
    /// 变更来源
    pub source: ConfigChangeSource,
}

/// Amp CLI 配置变更事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmpConfigChangeEvent {
    pub upstream_url: Option<String>,
    pub model_mappings_count: usize,
    /// 变更来源
    pub source: ConfigChangeSource,
}

/// Native Agent 配置变更事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeAgentChangeEvent {
    pub default_model: String,
    pub temperature: f32,
    pub max_tokens: u32,
    /// 变更来源
    pub source: ConfigChangeSource,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_serialization() {
        let event = ConfigChangeEvent::RoutingChanged(RoutingChangeEvent {
            default_provider: Some("kiro".to_string()),
            model_aliases_changed: false,
            model_aliases: None,
            source: ConfigChangeSource::ApiCall,
        });

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("RoutingChanged"));
        assert!(json.contains("kiro"));
    }

    #[test]
    fn test_event_type() {
        let event = ConfigChangeEvent::FullReload(FullReloadEvent {
            timestamp_ms: 12345,
            source: ConfigChangeSource::HotReload,
        });
        assert_eq!(event.event_type(), "FullReload");
    }
}
