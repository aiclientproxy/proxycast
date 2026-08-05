//! MCP 类型定义
//!
//! 本模块定义 MCP 协议相关的数据类型，包括：
//! - 服务器配置和状态
//! - 工具定义、调用和结果
//! - 提示词定义和结果
//! - 资源定义和内容
//! - 错误类型
//! - 错误类型

use crate::auth_status::McpServerAuthStatus;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

/// MCP 服务器未显式配置时使用的本地环境身份。
pub const DEFAULT_MCP_SERVER_ENVIRONMENT_ID: &str = "local";

// ============================================================================
// 服务器配置和状态
// ============================================================================

/// MCP 服务器传输类型。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum McpServerTransport {
    /// 本地 stdio 进程。
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        env: HashMap<String, String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cwd: Option<String>,
    },
    /// MCP streamable HTTP。
    StreamableHttp {
        url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bearer_token_env_var: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        http_headers: Option<HashMap<String, String>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        env_http_headers: Option<HashMap<String, String>>,
    },
}

/// MCP OAuth 客户端配置。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct McpServerOAuthConfig {
    #[serde(default, alias = "clientId", skip_serializing_if = "Option::is_none")]
    pub client_id: Option<String>,
}

impl Default for McpServerTransport {
    fn default() -> Self {
        Self::Stdio {
            command: String::new(),
            args: Vec::new(),
            env: HashMap::new(),
            cwd: None,
        }
    }
}

/// MCP 服务器配置。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct McpServerConfig {
    #[serde(flatten)]
    pub transport: McpServerTransport,
    /// MCP 服务器实际运行环境的显式身份，不从 cwd 推导。
    #[serde(default = "default_environment_id")]
    pub environment_id: String,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// 启动超时时间（秒）。
    #[serde(default = "default_timeout")]
    pub startup_timeout: u64,
    /// 工具调用超时时间（秒）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_timeout: Option<u64>,
    /// 显式允许的 MCP inner tool name 列表。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled_tools: Option<Vec<String>>,
    /// 显式禁用的 MCP inner tool name 列表。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub disabled_tools: Vec<String>,
    #[serde(default)]
    pub required: bool,
    #[serde(default)]
    pub supports_parallel_tool_calls: bool,
    /// OAuth scopes，仅供 streamable HTTP 动态授权使用。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scopes: Option<Vec<String>>,
    /// OAuth 客户端配置；运行时授权链路未接通前启动会 fail closed。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oauth: Option<McpServerOAuthConfig>,
    /// OAuth resource 参数（RFC 8707）；运行时授权链路未接通前启动会 fail closed。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oauth_resource: Option<String>,
}

fn default_timeout() -> u64 {
    30
}

fn default_enabled() -> bool {
    true
}

fn default_environment_id() -> String {
    DEFAULT_MCP_SERVER_ENVIRONMENT_ID.to_string()
}

impl McpServerConfig {
    pub fn command(&self) -> &str {
        match &self.transport {
            McpServerTransport::Stdio { command, .. } => command,
            McpServerTransport::StreamableHttp { .. } => "",
        }
    }

    pub fn args(&self) -> &[String] {
        match &self.transport {
            McpServerTransport::Stdio { args, .. } => args,
            McpServerTransport::StreamableHttp { .. } => &[],
        }
    }

    pub fn env(&self) -> &HashMap<String, String> {
        match &self.transport {
            McpServerTransport::Stdio { env, .. } => env,
            McpServerTransport::StreamableHttp { .. } => empty_env(),
        }
    }

    pub fn transport_kind(&self) -> &'static str {
        match self.transport {
            McpServerTransport::Stdio { .. } => "stdio",
            McpServerTransport::StreamableHttp { .. } => "streamable_http",
        }
    }

    /// 获取清洗后的工作目录（去除 `\0`、首尾空白，并展开 `~`）
    pub fn sanitized_cwd(&self) -> Option<PathBuf> {
        let cwd = match &self.transport {
            McpServerTransport::Stdio { cwd, .. } => cwd.as_deref()?,
            McpServerTransport::StreamableHttp { .. } => return None,
        };
        let cleaned = cwd.split('\0').next().unwrap_or_default().trim();
        if cleaned.is_empty() {
            return None;
        }

        if cleaned == "~" {
            return Some(dirs::home_dir().unwrap_or_else(|| PathBuf::from(cleaned)));
        }

        if cleaned.starts_with("~/") || cleaned.starts_with("~\\") {
            if let Some(home) = dirs::home_dir() {
                return Some(home.join(&cleaned[2..]));
            }
        }

        Some(PathBuf::from(cleaned))
    }

    pub fn startup_timeout_secs(&self) -> u64 {
        self.startup_timeout.clamp(1, 300)
    }

    pub fn tool_timeout_secs(&self) -> u64 {
        self.tool_timeout
            .unwrap_or(self.startup_timeout)
            .clamp(1, 300)
    }

    pub fn enabled_tool_set(&self) -> Option<HashSet<String>> {
        self.enabled_tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|tool| normalize_tool_name(tool))
                .filter(|tool| !tool.is_empty())
                .collect()
        })
    }

    pub fn disabled_tool_set(&self) -> HashSet<String> {
        self.disabled_tools
            .iter()
            .map(|tool| normalize_tool_name(tool))
            .filter(|tool| !tool.is_empty())
            .collect()
    }

    pub fn tool_is_enabled(&self, tool_name: &str) -> bool {
        let normalized = normalize_tool_name(tool_name);
        if normalized.is_empty() {
            return false;
        }
        if let Some(enabled_tools) = self.enabled_tool_set() {
            if !enabled_tools.contains(&normalized) {
                return false;
            }
        }
        !self.disabled_tool_set().contains(&normalized)
    }

    pub fn validate_static(&self) -> Result<(), String> {
        if self.environment_id.trim().is_empty() {
            return Err("environment_id must be a non-empty string".to_string());
        }
        if !self.enabled {
            return Ok(());
        }
        if self.startup_timeout == 0 || self.startup_timeout > 300 {
            return Err("startup_timeout must be between 1 and 300 seconds".to_string());
        }
        if let Some(tool_timeout) = self.tool_timeout {
            if tool_timeout == 0 || tool_timeout > 300 {
                return Err("tool_timeout must be between 1 and 300 seconds".to_string());
            }
        }
        match &self.transport {
            McpServerTransport::Stdio { command, .. } => {
                if command.trim().is_empty() {
                    return Err("stdio MCP requires command".to_string());
                }
                if self.has_oauth_settings() {
                    return Err(
                        "OAuth MCP settings are only supported for streamable_http transport"
                            .to_string(),
                    );
                }
                Ok(())
            }
            McpServerTransport::StreamableHttp { url, .. } => {
                if !(url.starts_with("http://") || url.starts_with("https://")) {
                    return Err(
                        "streamable_http MCP url must start with http:// or https://".to_string(),
                    );
                }
                Ok(())
            }
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        self.validate_static()?;
        if self.has_unsupported_oauth_runtime_settings() {
            return Err(
                "streamable_http MCP OAuth client_id/oauth_resource is not yet connected; use dynamic OAuth login without explicit client_id/oauth_resource or use bearer_token_env_var/HTTP headers"
                    .to_string(),
            );
        }
        Ok(())
    }
}

impl Default for McpServerConfig {
    fn default() -> Self {
        Self {
            transport: McpServerTransport::default(),
            environment_id: default_environment_id(),
            enabled: false,
            startup_timeout: default_timeout(),
            tool_timeout: None,
            enabled_tools: None,
            disabled_tools: Vec::new(),
            required: false,
            supports_parallel_tool_calls: false,
            scopes: None,
            oauth: None,
            oauth_resource: None,
        }
    }
}

fn normalize_tool_name(tool_name: &str) -> String {
    tool_name.trim().to_ascii_lowercase()
}

fn empty_env() -> &'static HashMap<String, String> {
    use std::sync::OnceLock;
    static EMPTY: OnceLock<HashMap<String, String>> = OnceLock::new();
    EMPTY.get_or_init(HashMap::new)
}

impl<'de> Deserialize<'de> for McpServerConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        Self::from_value(value).map_err(serde::de::Error::custom)
    }
}

impl McpServerConfig {
    pub fn from_value(value: serde_json::Value) -> Result<Self, String> {
        let Some(object) = value.as_object() else {
            return Err("MCP server config must be an object".to_string());
        };

        let transport_type = object
            .get("transport")
            .and_then(|value| value.as_str())
            .or_else(|| object.get("type").and_then(|value| value.as_str()));

        let has_inline_bearer_token =
            object.contains_key("bearer_token") || object.contains_key("bearerToken");

        let transport = match transport_type {
            Some("streamable_http") | Some("streamable-http") | Some("http") => {
                reject_inline_bearer_token(has_inline_bearer_token)?;
                McpServerTransport::StreamableHttp {
                    url: string_field(object, "url").ok_or("streamable_http MCP requires url")?,
                    bearer_token_env_var: string_field(object, "bearer_token_env_var")
                        .or_else(|| string_field(object, "bearerTokenEnvVar")),
                    http_headers: string_map_field(object, "http_headers")
                        .or_else(|| string_map_field(object, "httpHeaders")),
                    env_http_headers: string_map_field(object, "env_http_headers")
                        .or_else(|| string_map_field(object, "envHttpHeaders")),
                }
            }
            _ if object.contains_key("url") && !object.contains_key("command") => {
                reject_inline_bearer_token(has_inline_bearer_token)?;
                McpServerTransport::StreamableHttp {
                    url: string_field(object, "url").ok_or("streamable_http MCP requires url")?,
                    bearer_token_env_var: string_field(object, "bearer_token_env_var")
                        .or_else(|| string_field(object, "bearerTokenEnvVar")),
                    http_headers: string_map_field(object, "http_headers")
                        .or_else(|| string_map_field(object, "httpHeaders")),
                    env_http_headers: string_map_field(object, "env_http_headers")
                        .or_else(|| string_map_field(object, "envHttpHeaders")),
                }
            }
            _ => McpServerTransport::Stdio {
                command: string_field(object, "command").unwrap_or_default(),
                args: string_vec_field(object, "args"),
                env: string_map_field(object, "env").unwrap_or_default(),
                cwd: string_field(object, "cwd"),
            },
        };

        let startup_timeout = u64_field(object, "startup_timeout")
            .or_else(|| u64_field(object, "startupTimeout"))
            .or_else(|| u64_field(object, "startup_timeout_sec"))
            .or_else(|| u64_field(object, "startupTimeoutSec"))
            .or_else(|| u64_field(object, "timeout"))
            .unwrap_or_else(default_timeout);

        let environment_id = string_field(object, "environment_id")
            .or_else(|| string_field(object, "environmentId"))
            .unwrap_or_else(default_environment_id);
        if environment_id.trim().is_empty() {
            return Err("environment_id must be a non-empty string".to_string());
        }

        let config = Self {
            transport,
            environment_id,
            enabled: bool_field(object, "enabled").unwrap_or_else(default_enabled),
            startup_timeout,
            tool_timeout: u64_field(object, "tool_timeout")
                .or_else(|| u64_field(object, "toolTimeout"))
                .or_else(|| u64_field(object, "tool_timeout_sec"))
                .or_else(|| u64_field(object, "toolTimeoutSec")),
            enabled_tools: string_vec_field_optional(object, "enabled_tools")
                .or_else(|| string_vec_field_optional(object, "enabledTools")),
            disabled_tools: string_vec_field_optional(object, "disabled_tools")
                .or_else(|| string_vec_field_optional(object, "disabledTools"))
                .unwrap_or_default(),
            required: bool_field(object, "required").unwrap_or(false),
            supports_parallel_tool_calls: bool_field(object, "supports_parallel_tool_calls")
                .or_else(|| bool_field(object, "supportsParallelToolCalls"))
                .unwrap_or(false),
            scopes: required_string_vec_field_optional(object, "scopes")?,
            oauth: oauth_config_field(object)?,
            oauth_resource: optional_string_alias_field(object, "oauth_resource", "oauthResource")?,
        };
        config.validate_static()?;
        Ok(config)
    }
}

fn reject_inline_bearer_token(has_inline_bearer_token: bool) -> Result<(), String> {
    if has_inline_bearer_token {
        return Err(
            "streamable_http MCP inline bearer_token is not supported; use bearer_token_env_var"
                .to_string(),
        );
    }
    Ok(())
}

fn oauth_config_field(
    object: &serde_json::Map<String, serde_json::Value>,
) -> Result<Option<McpServerOAuthConfig>, String> {
    let Some(value) = object.get("oauth") else {
        return Ok(None);
    };
    let Some(oauth) = value.as_object() else {
        return Err("oauth must be an object".to_string());
    };
    for key in oauth.keys() {
        if key != "client_id" && key != "clientId" {
            return Err(format!("oauth contains unsupported field: {key}"));
        }
    }
    let client_id = optional_string_alias_field(oauth, "client_id", "clientId")?;
    Ok(Some(McpServerOAuthConfig { client_id }))
}

fn string_field(object: &serde_json::Map<String, serde_json::Value>, key: &str) -> Option<String> {
    object
        .get(key)
        .and_then(|value| value.as_str())
        .map(ToString::to_string)
}

fn optional_string_alias_field(
    object: &serde_json::Map<String, serde_json::Value>,
    snake_key: &str,
    camel_key: &str,
) -> Result<Option<String>, String> {
    let Some((key, value)) = object
        .get_key_value(snake_key)
        .or_else(|| object.get_key_value(camel_key))
    else {
        return Ok(None);
    };
    value
        .as_str()
        .map(ToString::to_string)
        .map(Some)
        .ok_or_else(|| format!("{key} must be a string"))
}

fn bool_field(object: &serde_json::Map<String, serde_json::Value>, key: &str) -> Option<bool> {
    object.get(key).and_then(|value| value.as_bool())
}

fn u64_field(object: &serde_json::Map<String, serde_json::Value>, key: &str) -> Option<u64> {
    object.get(key).and_then(|value| {
        value.as_u64().or_else(|| {
            value
                .as_f64()
                .and_then(|value| (value >= 0.0).then_some(value as u64))
        })
    })
}

fn string_vec_field(object: &serde_json::Map<String, serde_json::Value>, key: &str) -> Vec<String> {
    string_vec_field_optional(object, key).unwrap_or_default()
}

fn string_vec_field_optional(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<Vec<String>> {
    object.get(key).and_then(|value| {
        value.as_array().map(|values| {
            values
                .iter()
                .filter_map(|value| value.as_str().map(ToString::to_string))
                .collect()
        })
    })
}

fn required_string_vec_field_optional(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<Option<Vec<String>>, String> {
    let Some(value) = object.get(key) else {
        return Ok(None);
    };
    let Some(values) = value.as_array() else {
        return Err(format!("{key} must be an array of strings"));
    };
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(ToString::to_string)
                .ok_or_else(|| format!("{key} must be an array of strings"))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn string_map_field(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<HashMap<String, String>> {
    object.get(key).and_then(|value| {
        value.as_object().map(|values| {
            values
                .iter()
                .filter_map(|(key, value)| {
                    value.as_str().map(|value| (key.clone(), value.to_string()))
                })
                .collect()
        })
    })
}

/// MCP 服务器信息（包含运行状态）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerInfo {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub config: McpServerConfig,
    pub is_running: bool,
    pub server_info: Option<McpServerCapabilities>,
    pub enabled_lime: bool,
    pub enabled_claude: bool,
    pub enabled_codex: bool,
    pub enabled_gemini: bool,
}

/// MCP server 运行时状态快照。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct McpServerRuntimeStatus {
    pub name: String,
    pub transport: String,
    pub enabled: bool,
    pub is_running: bool,
    pub required: bool,
    pub supports_parallel_tool_calls: bool,
    pub startup_timeout: u64,
    pub tool_timeout: u64,
    pub enabled_tools: Option<Vec<String>>,
    pub disabled_tools: Vec<String>,
    pub server_info: Option<McpServerCapabilities>,
    pub auth_status: McpServerAuthStatus,
}

/// MCP 服务器能力
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerCapabilities {
    pub name: String,
    pub version: String,
    pub supports_tools: bool,
    pub supports_prompts: bool,
    pub supports_resources: bool,
}

// ============================================================================
// 工具类型
// ============================================================================

/// MCP 工具定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<serde_json::Value>,
    pub server_name: String,
    /// 是否延迟加载（不默认注入上下文）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deferred_loading: Option<bool>,
    /// 是否始终可见（即使 deferred_loading=true 也可见）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub always_visible: Option<bool>,
    /// 允许调用方（如 assistant/code_execution/tool_search）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_callers: Option<Vec<String>>,
    /// 工具输入示例
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_examples: Option<Vec<serde_json::Value>>,
    /// 标签（用于工具搜索）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
}

/// MCP 工具调用请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// MCP 工具调用结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolResult {
    pub content: Vec<McpContent>,
    #[serde(
        default,
        rename = "structuredContent",
        alias = "structured_content",
        skip_serializing_if = "Option::is_none"
    )]
    pub structured_content: Option<serde_json::Value>,
    pub is_error: bool,
}

/// MCP 内容类型
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum McpContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { data: String, mime_type: String },
    #[serde(rename = "resource")]
    Resource {
        uri: String,
        text: Option<String>,
        blob: Option<String>,
    },
}

// ============================================================================
// 提示词类型
// ============================================================================

/// MCP 提示词定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPromptDefinition {
    pub name: String,
    pub description: Option<String>,
    pub arguments: Vec<McpPromptArgument>,
    pub server_name: String,
}

/// MCP 提示词参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPromptArgument {
    pub name: String,
    pub description: Option<String>,
    pub required: bool,
}

/// MCP 提示词结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPromptResult {
    pub description: Option<String>,
    pub messages: Vec<McpPromptMessage>,
}

/// MCP 提示词消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPromptMessage {
    pub role: String,
    pub content: McpContent,
}

// ============================================================================
// 资源类型
// ============================================================================

/// MCP 资源定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResourceDefinition {
    pub uri: String,
    pub name: String,
    pub description: Option<String>,
    pub mime_type: Option<String>,
    pub server_name: String,
}

/// MCP 资源模板定义。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResourceTemplateDefinition {
    pub uri_template: String,
    pub name: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub mime_type: Option<String>,
    pub server_name: String,
}

/// MCP 资源内容
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResourceContent {
    pub uri: String,
    pub mime_type: Option<String>,
    pub text: Option<String>,
    pub blob: Option<String>,
    pub meta: Option<serde_json::Value>,
}

// ============================================================================
// 错误类型
// ============================================================================

/// MCP 错误类型
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("服务器配置不存在: {0}")]
    ConfigNotFound(String),

    #[error("服务器已在运行: {0}")]
    ServerAlreadyRunning(String),

    #[error("服务器未运行: {0}")]
    ServerNotRunning(String),

    #[error("无法启动服务器进程: {0}")]
    ProcessSpawnFailed(String),

    #[error("MCP 连接失败: {0}")]
    ConnectionFailed(String),

    #[error("工具不存在: {0}")]
    ToolNotFound(String),

    #[error("工具调用失败: {0}")]
    ToolCallFailed(String),

    #[error("操作超时")]
    Timeout,

    #[error("数据库错误: {0}")]
    DatabaseError(String),

    #[error("协议错误: {0}")]
    ProtocolError(String),

    #[error("配置错误: {0}")]
    ConfigError(String),
}

// ============================================================================
// 状态类型
// ============================================================================

use std::sync::Arc;
use tokio::sync::Mutex;

/// MCP 客户端管理器状态
///
/// 使用 Arc<Mutex<McpClientManager>> 包装，支持跨线程共享和异步访问。
pub type McpManagerState = Arc<Mutex<super::manager::McpClientManager>>;

#[cfg(test)]
mod tests {
    use super::{McpServerConfig, McpServerTransport, DEFAULT_MCP_SERVER_ENVIRONMENT_ID};
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn sample_config(cwd: Option<String>) -> McpServerConfig {
        McpServerConfig {
            transport: McpServerTransport::Stdio {
                command: "npx".to_string(),
                args: vec!["-y".to_string(), "some-server".to_string()],
                env: HashMap::new(),
                cwd,
            },
            environment_id: DEFAULT_MCP_SERVER_ENVIRONMENT_ID.to_string(),
            enabled: true,
            startup_timeout: 30,
            tool_timeout: None,
            enabled_tools: None,
            disabled_tools: Vec::new(),
            required: false,
            supports_parallel_tool_calls: false,
            scopes: None,
            oauth: None,
            oauth_resource: None,
        }
    }

    #[test]
    fn sanitized_cwd_should_strip_nul_suffix() {
        let config = sample_config(Some(" /tmp/demo\0ignored ".to_string()));
        assert_eq!(config.sanitized_cwd(), Some(PathBuf::from("/tmp/demo")));
    }

    #[test]
    fn sanitized_cwd_should_reject_empty_value() {
        let config = sample_config(Some(" \0 ".to_string()));
        assert!(config.sanitized_cwd().is_none());
    }

    #[test]
    fn parses_legacy_stdio_config() {
        let config = McpServerConfig::from_value(serde_json::json!({
            "command": "node",
            "args": ["server.js"],
            "env": { "A": "1" },
            "cwd": "~/project",
            "timeout": 42
        }))
        .unwrap();

        assert_eq!(config.transport_kind(), "stdio");
        assert_eq!(config.command(), "node");
        assert_eq!(config.args(), &["server.js".to_string()]);
        assert_eq!(config.startup_timeout, 42);
        assert_eq!(config.environment_id, DEFAULT_MCP_SERVER_ENVIRONMENT_ID);
    }

    #[test]
    fn parses_explicit_environment_identity_without_deriving_from_cwd() {
        let config = McpServerConfig::from_value(serde_json::json!({
            "command": "node",
            "cwd": "/tmp/other",
            "environmentId": "remote"
        }))
        .unwrap();

        assert_eq!(config.environment_id, "remote");
    }

    #[test]
    fn rejects_empty_environment_identity() {
        let error = McpServerConfig::from_value(serde_json::json!({
            "command": "node",
            "environment_id": "  "
        }))
        .unwrap_err();

        assert!(error.contains("environment_id"));
    }

    #[test]
    fn parses_streamable_http_config() {
        let config = McpServerConfig::from_value(serde_json::json!({
            "transport": "streamable_http",
            "url": "https://example.com/mcp",
            "bearer_token_env_var": "MCP_TOKEN",
            "tool_timeout_sec": 12,
            "enabled_tools": ["search"],
            "disabled_tools": ["delete"]
        }))
        .unwrap();

        assert_eq!(config.transport_kind(), "streamable_http");
        assert_eq!(config.tool_timeout_secs(), 12);
        assert!(config.tool_is_enabled("search"));
        assert!(!config.tool_is_enabled("delete"));
        assert!(!config.tool_is_enabled("other"));
    }

    #[test]
    fn parses_streamable_http_oauth_shape_but_runtime_validation_fails_closed() {
        let config = McpServerConfig::from_value(serde_json::json!({
            "transport": "streamable_http",
            "url": "https://example.com/mcp",
            "scopes": ["search.read"],
            "oauth": { "clientId": "lime-client" },
            "oauthResource": "https://example.com"
        }))
        .unwrap();

        assert_eq!(
            config.scopes.as_deref(),
            Some(&["search.read".to_string()][..])
        );
        assert_eq!(
            config
                .oauth
                .as_ref()
                .and_then(|oauth| oauth.client_id.as_deref()),
            Some("lime-client")
        );
        assert_eq!(
            config.oauth_resource.as_deref(),
            Some("https://example.com")
        );

        let error = config.validate().unwrap_err();
        assert!(error.contains("client_id/oauth_resource"));

        let auth_status = config.auth_status();
        assert_eq!(auth_status.mode, "oauth");
        assert!(!auth_status.available);
        assert_eq!(
            auth_status.reason_code.as_deref(),
            Some("oauth_runtime_not_implemented")
        );
    }

    #[test]
    fn parses_dynamic_streamable_http_oauth_shape_and_validation_allows_login() {
        let config = McpServerConfig::from_value(serde_json::json!({
            "transport": "streamable_http",
            "url": "https://example.com/mcp",
            "scopes": ["search.read"]
        }))
        .unwrap();

        assert!(config.validate().is_ok());
        let auth_status = config.auth_status();
        assert_eq!(auth_status.mode, "oauth");
        assert!(auth_status.available);
        assert_eq!(
            auth_status.reason_code.as_deref(),
            Some("oauth_login_required")
        );
        let plan = auth_status.action_plan.expect("oauth should expose plan");
        assert_eq!(plan.kind, "oauth_login");
        assert_eq!(plan.state, "login_required");
        assert_eq!(
            plan.required_runtime.as_deref(),
            Some("mcp_server_oauth_login")
        );
    }

    #[test]
    fn rejects_oauth_shape_on_stdio_transport() {
        let error = McpServerConfig::from_value(serde_json::json!({
            "transport": "stdio",
            "command": "node",
            "oauth": { "client_id": "lime-client" }
        }))
        .unwrap_err();

        assert!(error.contains("only supported for streamable_http"));
    }

    #[test]
    fn rejects_inline_bearer_token() {
        let error = McpServerConfig::from_value(serde_json::json!({
            "transport": "streamable_http",
            "url": "https://example.com/mcp",
            "bearer_token": "secret"
        }))
        .unwrap_err();

        assert!(error.contains("bearer_token_env_var"));
    }

    #[test]
    fn rejects_invalid_oauth_shapes() {
        let scopes_error = McpServerConfig::from_value(serde_json::json!({
            "transport": "streamable_http",
            "url": "https://example.com/mcp",
            "scopes": ["ok", 1]
        }))
        .unwrap_err();
        assert!(scopes_error.contains("scopes"));

        let client_error = McpServerConfig::from_value(serde_json::json!({
            "transport": "streamable_http",
            "url": "https://example.com/mcp",
            "oauth": { "client_id": 1 }
        }))
        .unwrap_err();
        assert!(client_error.contains("client_id"));

        let resource_error = McpServerConfig::from_value(serde_json::json!({
            "transport": "streamable_http",
            "url": "https://example.com/mcp",
            "oauth_resource": 1
        }))
        .unwrap_err();
        assert!(resource_error.contains("oauth_resource"));
    }

    #[test]
    fn reports_static_header_auth_status_for_streamable_http_secrets() {
        let config = McpServerConfig::from_value(serde_json::json!({
            "transport": "streamable_http",
            "url": "https://example.com/mcp",
            "bearer_token_env_var": "MCP_TOKEN"
        }))
        .unwrap();

        let auth_status = config.auth_status();
        assert_eq!(auth_status.mode, "static_headers");
        assert!(auth_status.available);
        assert!(auth_status.reason_code.is_none());
    }

    #[test]
    fn reports_none_auth_status_for_unauthenticated_servers() {
        let config = McpServerConfig::from_value(serde_json::json!({
            "transport": "streamable_http",
            "url": "https://example.com/mcp"
        }))
        .unwrap();

        let auth_status = config.auth_status();
        assert_eq!(auth_status.mode, "none");
        assert!(auth_status.available);
        assert!(auth_status.reason_code.is_none());
    }

    #[test]
    fn parses_streamable_http_headers_in_snake_and_camel_case() {
        let snake = McpServerConfig::from_value(serde_json::json!({
            "transport": "streamable_http",
            "url": "https://example.com/mcp",
            "http_headers": { "X-Foo": "bar" },
            "env_http_headers": { "X-Token": "MCP_TOKEN" }
        }))
        .unwrap();
        let McpServerTransport::StreamableHttp {
            http_headers,
            env_http_headers,
            ..
        } = snake.transport
        else {
            panic!("expected streamable_http transport");
        };
        assert_eq!(http_headers.unwrap().get("X-Foo").unwrap(), "bar");
        assert_eq!(
            env_http_headers.unwrap().get("X-Token").unwrap(),
            "MCP_TOKEN"
        );

        let camel = McpServerConfig::from_value(serde_json::json!({
            "transport": "streamable-http",
            "url": "https://example.com/mcp",
            "httpHeaders": { "X-Foo": "bar" },
            "envHttpHeaders": { "X-Token": "MCP_TOKEN" }
        }))
        .unwrap();
        let McpServerTransport::StreamableHttp {
            http_headers,
            env_http_headers,
            ..
        } = camel.transport
        else {
            panic!("expected streamable_http transport");
        };
        assert_eq!(http_headers.unwrap().get("X-Foo").unwrap(), "bar");
        assert_eq!(
            env_http_headers.unwrap().get("X-Token").unwrap(),
            "MCP_TOKEN"
        );
    }

    #[test]
    fn validates_streamable_http_url() {
        let error = McpServerConfig::from_value(serde_json::json!({
            "transport": "streamable_http",
            "url": "file:///tmp/mcp"
        }))
        .unwrap_err();

        assert!(error.contains("http:// or https://"));
    }
}
