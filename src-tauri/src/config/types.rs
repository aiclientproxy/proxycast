//! 配置类型定义
//!
//! 定义 ProxyCast 的配置结构，支持 YAML 和 JSON 序列化/反序列化
//! 保持与旧版 JSON 配置的向后兼容性

use crate::injection::{InjectionMode, InjectionRule};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============ 凭证池配置类型 ============

/// 凭证池配置
///
/// 管理多个 Provider 的多个凭证，支持负载均衡
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct CredentialPoolConfig {
    /// Kiro 凭证列表（OAuth）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub kiro: Vec<CredentialEntry>,
    /// Gemini 凭证列表（OAuth）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gemini: Vec<CredentialEntry>,
    /// Qwen 凭证列表（OAuth）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub qwen: Vec<CredentialEntry>,
    /// OpenAI 凭证列表（API Key）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub openai: Vec<ApiKeyEntry>,
    /// Claude 凭证列表（API Key）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub claude: Vec<ApiKeyEntry>,
    /// Gemini API Key 凭证列表（多账号负载均衡）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gemini_api_keys: Vec<GeminiApiKeyEntry>,
    /// Vertex AI 凭证列表
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vertex_api_keys: Vec<VertexApiKeyEntry>,
    /// Codex OAuth 凭证列表
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub codex: Vec<CredentialEntry>,
    /// iFlow 凭证列表
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub iflow: Vec<IFlowCredentialEntry>,
}

/// Gemini API Key 凭证条目
///
/// 用于 Gemini API Key 多账号负载均衡
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GeminiApiKeyEntry {
    /// 凭证 ID
    pub id: String,
    /// API Key
    pub api_key: String,
    /// 自定义 Base URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// 单独的代理 URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy_url: Option<String>,
    /// 排除的模型列表（支持通配符）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub excluded_models: Vec<String>,
    /// 是否禁用
    #[serde(default)]
    pub disabled: bool,
}

/// Vertex AI 模型别名映射
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VertexModelAlias {
    /// 上游模型名称
    pub name: String,
    /// 客户端可见的别名
    pub alias: String,
}

/// Vertex AI 凭证条目
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VertexApiKeyEntry {
    /// 凭证 ID
    pub id: String,
    /// API Key
    pub api_key: String,
    /// Base URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// 模型别名映射
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<VertexModelAlias>,
    /// 单独的代理 URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy_url: Option<String>,
    /// 是否禁用
    #[serde(default)]
    pub disabled: bool,
}

/// iFlow 凭证条目
///
/// 支持 OAuth 和 Cookie 两种认证方式
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IFlowCredentialEntry {
    /// 凭证 ID
    pub id: String,
    /// Token 文件路径（OAuth 模式）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_file: Option<String>,
    /// 认证类型：oauth 或 cookie
    #[serde(default = "default_auth_type")]
    pub auth_type: String,
    /// Cookie 字符串（Cookie 模式）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cookies: Option<String>,
    /// 单独的代理 URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy_url: Option<String>,
    /// 是否禁用
    #[serde(default)]
    pub disabled: bool,
}

fn default_auth_type() -> String {
    "oauth".to_string()
}

/// OAuth 凭证条目
///
/// 用于 Kiro、Gemini、Qwen 等 OAuth 认证的 Provider
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CredentialEntry {
    /// 凭证 ID
    pub id: String,
    /// Token 文件路径（相对于 auth_dir）
    pub token_file: String,
    /// 是否禁用
    #[serde(default)]
    pub disabled: bool,
    /// 单独的代理 URL（覆盖全局代理）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy_url: Option<String>,
}

/// API Key 凭证条目
///
/// 用于 OpenAI、Claude 等 API Key 认证的 Provider
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiKeyEntry {
    /// 凭证 ID
    pub id: String,
    /// API Key
    pub api_key: String,
    /// 自定义 Base URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// 是否禁用
    #[serde(default)]
    pub disabled: bool,
    /// 单独的代理 URL（覆盖全局代理）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy_url: Option<String>,
}

/// 默认 auth_dir 路径
fn default_auth_dir() -> String {
    "~/.proxycast/auth".to_string()
}

/// 端点 Provider 配置
///
/// 允许为不同的客户端端点配置不同的 Provider
/// 例如：Cursor 使用 Qwen，Claude Code 使用 Kiro，Codex 使用 Codex
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct EndpointProvidersConfig {
    /// Cursor 客户端使用的 Provider
    /// 如果为空，则使用 default_provider
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    /// Claude Code 客户端使用的 Provider
    /// 如果为空，则使用 default_provider
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claude_code: Option<String>,
    /// Codex 客户端使用的 Provider
    /// 如果为空，则使用 default_provider
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codex: Option<String>,
    /// Windsurf 客户端使用的 Provider
    /// 如果为空，则使用 default_provider
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub windsurf: Option<String>,
    /// Kiro 客户端使用的 Provider
    /// 如果为空，则使用 default_provider
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kiro: Option<String>,
    /// 其他客户端使用的 Provider
    /// 如果为空，则使用 default_provider
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub other: Option<String>,
}

impl EndpointProvidersConfig {
    /// 根据客户端类型获取配置的 Provider
    ///
    /// # 参数
    /// - `client_type`: 客户端类型的配置键名（cursor, claude_code, codex, windsurf, kiro, other）
    ///
    /// # 返回
    /// 如果配置了对应的 Provider，返回 Some(&String)；否则返回 None
    pub fn get_provider(&self, client_type: &str) -> Option<&String> {
        match client_type {
            "cursor" => self.cursor.as_ref(),
            "claude_code" => self.claude_code.as_ref(),
            "codex" => self.codex.as_ref(),
            "windsurf" => self.windsurf.as_ref(),
            "kiro" => self.kiro.as_ref(),
            "other" => self.other.as_ref(),
            _ => None,
        }
    }

    /// 设置客户端类型的 Provider
    ///
    /// # 参数
    /// - `client_type`: 客户端类型的配置键名（cursor, claude_code, codex, windsurf, kiro, other）
    /// - `provider`: 要设置的 Provider 名称，None 或空字符串表示清除配置
    ///
    /// # 返回
    /// 如果客户端类型有效，返回 true；否则返回 false
    pub fn set_provider(&mut self, client_type: &str, provider: Option<String>) -> bool {
        let provider = provider.filter(|p| !p.is_empty());
        match client_type {
            "cursor" => {
                self.cursor = provider;
                true
            }
            "claude_code" => {
                self.claude_code = provider;
                true
            }
            "codex" => {
                self.codex = provider;
                true
            }
            "windsurf" => {
                self.windsurf = provider;
                true
            }
            "kiro" => {
                self.kiro = provider;
                true
            }
            "other" => {
                self.other = provider;
                true
            }
            _ => false,
        }
    }
}

/// 主配置结构
///
/// 支持两种格式：
/// - 旧版 JSON 格式：`default_provider` 在顶层
/// - 新版 YAML 格式：`default_provider` 在 `routing` 中
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Config {
    /// 服务器配置
    #[serde(default)]
    pub server: ServerConfig,
    /// Provider 配置
    #[serde(default)]
    pub providers: ProvidersConfig,
    /// 默认 Provider（向后兼容旧版 JSON 配置）
    #[serde(default = "default_provider")]
    pub default_provider: String,
    /// 路由配置（新版 YAML 配置）
    #[serde(default)]
    pub routing: RoutingConfig,
    /// 重试配置
    #[serde(default)]
    pub retry: RetrySettings,
    /// 日志配置
    #[serde(default)]
    pub logging: LoggingConfig,
    /// 参数注入配置
    #[serde(default)]
    pub injection: InjectionSettings,
    /// 认证目录路径（存储 OAuth Token 文件，支持 ~ 展开）
    #[serde(default = "default_auth_dir")]
    pub auth_dir: String,
    /// 凭证池配置
    #[serde(default)]
    pub credential_pool: CredentialPoolConfig,
    /// 远程管理配置
    #[serde(default)]
    pub remote_management: RemoteManagementConfig,
    /// 配额超限配置
    #[serde(default)]
    pub quota_exceeded: QuotaExceededConfig,
    /// 全局代理 URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy_url: Option<String>,
    /// Amp CLI 配置
    #[serde(default)]
    pub ampcode: AmpConfig,
    /// 端点 Provider 配置
    /// 允许为不同的客户端端点（CC/Codex）配置不同的 Provider
    #[serde(default)]
    pub endpoint_providers: EndpointProvidersConfig,
    /// 关闭时最小化到托盘（而不是退出应用）
    #[serde(default = "default_minimize_to_tray")]
    pub minimize_to_tray: bool,
}

fn default_minimize_to_tray() -> bool {
    true
}

/// 服务器配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServerConfig {
    /// 监听地址
    #[serde(default = "default_host")]
    pub host: String,
    /// 监听端口
    #[serde(default = "default_port")]
    pub port: u16,
    /// API 密钥
    #[serde(default = "default_api_key")]
    pub api_key: String,
    /// TLS 配置
    #[serde(default)]
    pub tls: TlsConfig,
}

/// TLS 配置
///
/// 用于启用 HTTPS 支持
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct TlsConfig {
    /// 是否启用 TLS
    #[serde(default)]
    pub enable: bool,
    /// 证书文件路径
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cert_path: Option<String>,
    /// 私钥文件路径
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_path: Option<String>,
}

/// 远程管理配置
///
/// 用于配置远程管理 API 的访问控制
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct RemoteManagementConfig {
    /// 是否允许远程访问（非 localhost）
    #[serde(default)]
    pub allow_remote: bool,
    /// 管理 API 密钥（为空时禁用管理 API）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secret_key: Option<String>,
    /// 是否禁用控制面板
    #[serde(default)]
    pub disable_control_panel: bool,
}

/// 配额超限配置
///
/// 用于配置配额超限时的自动切换策略
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QuotaExceededConfig {
    /// 是否自动切换到下一个凭证
    #[serde(default = "default_switch_project")]
    pub switch_project: bool,
    /// 是否尝试使用预览模型
    #[serde(default = "default_switch_preview_model")]
    pub switch_preview_model: bool,
    /// 冷却时间（秒）
    #[serde(default = "default_cooldown_seconds")]
    pub cooldown_seconds: u64,
}

fn default_switch_project() -> bool {
    true
}

fn default_switch_preview_model() -> bool {
    true
}

fn default_cooldown_seconds() -> u64 {
    300
}

impl Default for QuotaExceededConfig {
    fn default() -> Self {
        Self {
            switch_project: default_switch_project(),
            switch_preview_model: default_switch_preview_model(),
            cooldown_seconds: default_cooldown_seconds(),
        }
    }
}

/// Amp CLI 模型映射
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AmpModelMapping {
    /// 源模型名称
    pub from: String,
    /// 目标模型名称
    pub to: String,
}

/// Amp CLI 配置
///
/// 用于 Amp CLI 集成
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct AmpConfig {
    /// 上游 URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream_url: Option<String>,
    /// 模型映射列表
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub model_mappings: Vec<AmpModelMapping>,
    /// 是否限制管理端点只能从 localhost 访问
    #[serde(default)]
    pub restrict_management_to_localhost: bool,
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}

fn default_port() -> u16 {
    8999
}

pub const DEFAULT_API_KEY: &str = "proxy_cast";

fn default_api_key() -> String {
    DEFAULT_API_KEY.to_string()
}

/// 生成安全 API Key（32 字节随机）
pub fn generate_secure_api_key() -> String {
    use rand::distributions::Alphanumeric;
    use rand::Rng;

    let token: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(32)
        .map(char::from)
        .collect();
    format!("pc_{token}")
}

/// 是否为默认 API Key
pub fn is_default_api_key(api_key: &str) -> bool {
    api_key == DEFAULT_API_KEY
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            api_key: default_api_key(),
            tls: TlsConfig::default(),
        }
    }
}

/// Provider 配置集合
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProvidersConfig {
    /// Kiro Provider 配置
    #[serde(default)]
    pub kiro: ProviderConfig,
    /// Gemini Provider 配置
    #[serde(default)]
    pub gemini: ProviderConfig,
    /// Qwen Provider 配置
    #[serde(default)]
    pub qwen: ProviderConfig,
    /// OpenAI 自定义 Provider 配置
    #[serde(default)]
    pub openai: CustomProviderConfig,
    /// Claude 自定义 Provider 配置
    #[serde(default)]
    pub claude: CustomProviderConfig,
}

impl Default for ProvidersConfig {
    fn default() -> Self {
        Self {
            kiro: ProviderConfig {
                enabled: true,
                credentials_path: Some("~/.aws/sso/cache/kiro-auth-token.json".to_string()),
                region: Some("us-east-1".to_string()),
                project_id: None,
            },
            gemini: ProviderConfig {
                enabled: false,
                credentials_path: Some("~/.gemini/oauth_creds.json".to_string()),
                region: None,
                project_id: None,
            },
            qwen: ProviderConfig {
                enabled: false,
                credentials_path: Some("~/.qwen/oauth_creds.json".to_string()),
                region: None,
                project_id: None,
            },
            openai: CustomProviderConfig {
                enabled: false,
                api_key: None,
                base_url: Some("https://api.openai.com/v1".to_string()),
            },
            claude: CustomProviderConfig {
                enabled: false,
                api_key: None,
                base_url: Some("https://api.anthropic.com".to_string()),
            },
        }
    }
}

/// OAuth Provider 配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ProviderConfig {
    /// 是否启用
    #[serde(default)]
    pub enabled: bool,
    /// 凭证文件路径
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credentials_path: Option<String>,
    /// 区域
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// 项目 ID
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
}

/// 自定义 Provider 配置（API Key 方式）
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct CustomProviderConfig {
    /// 是否启用
    #[serde(default)]
    pub enabled: bool,
    /// API 密钥
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// 基础 URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
}

/// 路由配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RoutingConfig {
    /// 默认 Provider
    #[serde(default = "default_provider")]
    pub default_provider: String,
    /// 路由规则
    #[serde(default)]
    pub rules: Vec<RoutingRuleConfig>,
    /// 模型别名映射
    #[serde(default)]
    pub model_aliases: HashMap<String, String>,
    /// 排除列表（按 Provider）
    #[serde(default)]
    pub exclusions: HashMap<String, Vec<String>>,
}

fn default_provider() -> String {
    "kiro".to_string()
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            default_provider: default_provider(),
            rules: default_routing_rules(),
            model_aliases: HashMap::new(),
            exclusions: HashMap::new(),
        }
    }
}

/// 默认路由规则
///
/// 为常见的模型模式提供默认路由：
/// - `gemini-*` → Antigravity (Antigravity 支持 Gemini 系列模型)
/// - `claude-*` → Kiro (默认使用 Kiro 处理 Claude 模型)
fn default_routing_rules() -> Vec<RoutingRuleConfig> {
    vec![
        // Gemini 模型路由到 Antigravity
        RoutingRuleConfig {
            pattern: "gemini-*".to_string(),
            provider: "antigravity".to_string(),
            priority: 10,
        },
        // Claude 模型路由到 Kiro
        RoutingRuleConfig {
            pattern: "claude-*".to_string(),
            provider: "kiro".to_string(),
            priority: 10,
        },
    ]
}

/// 路由规则配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RoutingRuleConfig {
    /// 模型模式（支持通配符）
    pub pattern: String,
    /// 目标 Provider
    pub provider: String,
    /// 优先级（数字越小优先级越高）
    #[serde(default = "default_priority")]
    pub priority: i32,
}

fn default_priority() -> i32 {
    100
}

/// 重试配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetrySettings {
    /// 最大重试次数
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// 基础延迟（毫秒）
    #[serde(default = "default_base_delay_ms")]
    pub base_delay_ms: u64,
    /// 最大延迟（毫秒）
    #[serde(default = "default_max_delay_ms")]
    pub max_delay_ms: u64,
    /// 是否自动切换 Provider
    #[serde(default = "default_auto_switch")]
    pub auto_switch_provider: bool,
}

fn default_max_retries() -> u32 {
    3
}

fn default_base_delay_ms() -> u64 {
    1000
}

fn default_max_delay_ms() -> u64 {
    30000
}

fn default_auto_switch() -> bool {
    true
}

impl Default for RetrySettings {
    fn default() -> Self {
        Self {
            max_retries: default_max_retries(),
            base_delay_ms: default_base_delay_ms(),
            max_delay_ms: default_max_delay_ms(),
            auto_switch_provider: default_auto_switch(),
        }
    }
}

/// 日志配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoggingConfig {
    /// 是否启用日志
    #[serde(default = "default_logging_enabled")]
    pub enabled: bool,
    /// 日志级别
    #[serde(default = "default_log_level")]
    pub level: String,
    /// 日志保留天数
    #[serde(default = "default_retention_days")]
    pub retention_days: u32,
    /// 是否包含请求体
    #[serde(default)]
    pub include_request_body: bool,
}

fn default_logging_enabled() -> bool {
    true
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_retention_days() -> u32 {
    7
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            enabled: default_logging_enabled(),
            level: default_log_level(),
            retention_days: default_retention_days(),
            include_request_body: false,
        }
    }
}

/// 参数注入配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InjectionSettings {
    /// 是否启用参数注入
    #[serde(default = "default_injection_enabled")]
    pub enabled: bool,
    /// 注入规则列表
    #[serde(default)]
    pub rules: Vec<InjectionRuleConfig>,
}

fn default_injection_enabled() -> bool {
    false
}

impl Default for InjectionSettings {
    fn default() -> Self {
        Self {
            enabled: default_injection_enabled(),
            rules: Vec::new(),
        }
    }
}

/// 注入规则配置（用于 YAML/JSON 序列化）
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InjectionRuleConfig {
    /// 规则 ID
    pub id: String,
    /// 模型匹配模式（支持通配符）
    pub pattern: String,
    /// 要注入的参数
    pub parameters: serde_json::Value,
    /// 注入模式
    #[serde(default)]
    pub mode: InjectionMode,
    /// 优先级（数字越小优先级越高）
    #[serde(default = "default_priority")]
    pub priority: i32,
    /// 是否启用
    #[serde(default = "default_rule_enabled")]
    pub enabled: bool,
}

fn default_rule_enabled() -> bool {
    true
}

impl From<InjectionRuleConfig> for InjectionRule {
    fn from(config: InjectionRuleConfig) -> Self {
        let mut rule = InjectionRule::new(&config.id, &config.pattern, config.parameters);
        rule.mode = config.mode;
        rule.priority = config.priority;
        rule.enabled = config.enabled;
        rule
    }
}

impl From<&InjectionRule> for InjectionRuleConfig {
    fn from(rule: &InjectionRule) -> Self {
        Self {
            id: rule.id.clone(),
            pattern: rule.pattern.clone(),
            parameters: rule.parameters.clone(),
            mode: rule.mode,
            priority: rule.priority,
            enabled: rule.enabled,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            providers: ProvidersConfig::default(),
            default_provider: default_provider(),
            routing: RoutingConfig::default(),
            retry: RetrySettings::default(),
            logging: LoggingConfig::default(),
            injection: InjectionSettings::default(),
            auth_dir: default_auth_dir(),
            credential_pool: CredentialPoolConfig::default(),
            remote_management: RemoteManagementConfig::default(),
            quota_exceeded: QuotaExceededConfig::default(),
            proxy_url: None,
            ampcode: AmpConfig::default(),
            endpoint_providers: EndpointProvidersConfig::default(),
            minimize_to_tray: default_minimize_to_tray(),
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 8999);
        assert_eq!(config.server.api_key, "proxy_cast");
        assert!(config.providers.kiro.enabled);
        assert!(!config.providers.gemini.enabled);
        assert_eq!(config.default_provider, "kiro");
        assert_eq!(config.routing.default_provider, "kiro");
        assert_eq!(config.retry.max_retries, 3);
        assert!(config.logging.enabled);
        assert!(!config.injection.enabled);
        assert!(config.injection.rules.is_empty());
        // 新增字段测试
        assert_eq!(config.auth_dir, "~/.proxycast/auth");
        assert!(config.credential_pool.kiro.is_empty());
        assert!(config.credential_pool.openai.is_empty());
    }

    #[test]
    fn test_credential_pool_config_default() {
        let pool = CredentialPoolConfig::default();
        assert!(pool.kiro.is_empty());
        assert!(pool.gemini.is_empty());
        assert!(pool.qwen.is_empty());
        assert!(pool.openai.is_empty());
        assert!(pool.claude.is_empty());
    }

    #[test]
    fn test_credential_entry_serialization() {
        let entry = CredentialEntry {
            id: "kiro-main".to_string(),
            token_file: "kiro/main-token.json".to_string(),
            disabled: false,
            proxy_url: None,
        };
        let yaml = serde_yaml::to_string(&entry).unwrap();
        assert!(yaml.contains("id: kiro-main"));
        assert!(yaml.contains("token_file: kiro/main-token.json"));

        let parsed: CredentialEntry = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed, entry);
    }

    #[test]
    fn test_api_key_entry_serialization() {
        let entry = ApiKeyEntry {
            id: "openai-main".to_string(),
            api_key: "sk-test-key".to_string(),
            base_url: Some("https://api.openai.com/v1".to_string()),
            disabled: false,
            proxy_url: None,
        };
        let yaml = serde_yaml::to_string(&entry).unwrap();
        assert!(yaml.contains("id: openai-main"));
        assert!(yaml.contains("api_key: sk-test-key"));

        let parsed: ApiKeyEntry = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed, entry);
    }

    #[test]
    fn test_api_key_entry_without_base_url() {
        let entry = ApiKeyEntry {
            id: "claude-main".to_string(),
            api_key: "sk-ant-test".to_string(),
            base_url: None,
            disabled: true,
            proxy_url: None,
        };
        let yaml = serde_yaml::to_string(&entry).unwrap();
        // base_url should be skipped when None
        assert!(!yaml.contains("base_url"));
        assert!(yaml.contains("disabled: true"));

        let parsed: ApiKeyEntry = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed, entry);
    }

    #[test]
    fn test_credential_pool_config_serialization() {
        let pool = CredentialPoolConfig {
            kiro: vec![CredentialEntry {
                id: "kiro-1".to_string(),
                token_file: "kiro/token-1.json".to_string(),
                disabled: false,
                proxy_url: None,
            }],
            gemini: vec![],
            qwen: vec![],
            openai: vec![ApiKeyEntry {
                id: "openai-1".to_string(),
                api_key: "sk-xxx".to_string(),
                base_url: None,
                disabled: false,
                proxy_url: None,
            }],
            claude: vec![],
            gemini_api_keys: vec![],
            vertex_api_keys: vec![],
            codex: vec![],
            iflow: vec![],
        };

        let yaml = serde_yaml::to_string(&pool).unwrap();
        // Empty vecs should be skipped
        assert!(!yaml.contains("gemini"));
        assert!(!yaml.contains("qwen"));
        assert!(!yaml.contains("claude"));
        assert!(yaml.contains("kiro"));
        assert!(yaml.contains("openai"));

        let parsed: CredentialPoolConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed, pool);
    }

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 8999);
        assert_eq!(config.api_key, "proxy_cast");
    }

    #[test]
    fn test_retry_settings_default() {
        let config = RetrySettings::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.base_delay_ms, 1000);
        assert_eq!(config.max_delay_ms, 30000);
        assert!(config.auto_switch_provider);
    }

    #[test]
    fn test_logging_config_default() {
        let config = LoggingConfig::default();
        assert!(config.enabled);
        assert_eq!(config.level, "info");
        assert_eq!(config.retention_days, 7);
        assert!(!config.include_request_body);
    }

    #[test]
    fn test_routing_config_default() {
        let config = RoutingConfig::default();
        assert_eq!(config.default_provider, "kiro");
        // 默认包含 gemini-* 和 claude-* 的路由规则
        assert_eq!(config.rules.len(), 2);
        assert_eq!(config.rules[0].pattern, "gemini-*");
        assert_eq!(config.rules[0].provider, "antigravity");
        assert_eq!(config.rules[1].pattern, "claude-*");
        assert_eq!(config.rules[1].provider, "kiro");
        assert!(config.model_aliases.is_empty());
        assert!(config.exclusions.is_empty());
    }

    #[test]
    fn test_endpoint_providers_config_default() {
        let config = EndpointProvidersConfig::default();
        assert!(config.cursor.is_none());
        assert!(config.claude_code.is_none());
        assert!(config.codex.is_none());
        assert!(config.windsurf.is_none());
        assert!(config.kiro.is_none());
        assert!(config.other.is_none());
    }

    #[test]
    fn test_endpoint_providers_config_get_provider() {
        let config = EndpointProvidersConfig {
            cursor: Some("qwen".to_string()),
            claude_code: Some("kiro".to_string()),
            codex: Some("codex".to_string()),
            windsurf: None,
            kiro: Some("gemini".to_string()),
            other: None,
        };

        assert_eq!(config.get_provider("cursor"), Some(&"qwen".to_string()));
        assert_eq!(
            config.get_provider("claude_code"),
            Some(&"kiro".to_string())
        );
        assert_eq!(config.get_provider("codex"), Some(&"codex".to_string()));
        assert_eq!(config.get_provider("windsurf"), None);
        assert_eq!(config.get_provider("kiro"), Some(&"gemini".to_string()));
        assert_eq!(config.get_provider("other"), None);
        assert_eq!(config.get_provider("invalid"), None);
    }

    #[test]
    fn test_endpoint_providers_config_set_provider() {
        let mut config = EndpointProvidersConfig::default();

        // 设置有效的客户端类型
        assert!(config.set_provider("cursor", Some("qwen".to_string())));
        assert_eq!(config.cursor, Some("qwen".to_string()));

        assert!(config.set_provider("claude_code", Some("kiro".to_string())));
        assert_eq!(config.claude_code, Some("kiro".to_string()));

        assert!(config.set_provider("codex", Some("codex".to_string())));
        assert_eq!(config.codex, Some("codex".to_string()));

        assert!(config.set_provider("windsurf", Some("gemini".to_string())));
        assert_eq!(config.windsurf, Some("gemini".to_string()));

        assert!(config.set_provider("kiro", Some("openai".to_string())));
        assert_eq!(config.kiro, Some("openai".to_string()));

        assert!(config.set_provider("other", Some("claude".to_string())));
        assert_eq!(config.other, Some("claude".to_string()));

        // 设置无效的客户端类型
        assert!(!config.set_provider("invalid", Some("test".to_string())));
    }

    #[test]
    fn test_endpoint_providers_config_set_provider_clear() {
        let mut config = EndpointProvidersConfig {
            cursor: Some("qwen".to_string()),
            claude_code: Some("kiro".to_string()),
            codex: None,
            windsurf: None,
            kiro: None,
            other: None,
        };

        // 使用 None 清除配置
        assert!(config.set_provider("cursor", None));
        assert_eq!(config.cursor, None);

        // 使用空字符串清除配置
        assert!(config.set_provider("claude_code", Some("".to_string())));
        assert_eq!(config.claude_code, None);
    }

    #[test]
    fn test_endpoint_providers_config_serialization() {
        let config = EndpointProvidersConfig {
            cursor: Some("qwen".to_string()),
            claude_code: Some("kiro".to_string()),
            codex: None,
            windsurf: None,
            kiro: None,
            other: None,
        };

        let yaml = serde_yaml::to_string(&config).unwrap();
        assert!(yaml.contains("cursor: qwen"));
        assert!(yaml.contains("claude_code: kiro"));
        // None 值应该被跳过
        assert!(!yaml.contains("codex"));
        assert!(!yaml.contains("windsurf"));
        assert!(!yaml.contains("kiro:"));
        assert!(!yaml.contains("other"));

        let parsed: EndpointProvidersConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed, config);
    }

    #[test]
    fn test_endpoint_providers_config_json_serialization() {
        let config = EndpointProvidersConfig {
            cursor: Some("qwen".to_string()),
            claude_code: None,
            codex: Some("codex".to_string()),
            windsurf: None,
            kiro: None,
            other: Some("openai".to_string()),
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: EndpointProvidersConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, config);
    }
}
