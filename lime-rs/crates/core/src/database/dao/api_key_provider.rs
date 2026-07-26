//! API Key Provider 数据访问对象
//!
//! 提供 API Key Provider 的 CRUD 操作。
//!
//! **Feature: provider-ui-refactor**
//! **Validates: Requirements 9.1**

use crate::provider_prompt_cache_support::is_known_automatic_anthropic_compatible_host;
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

// ============================================================================
// 数据模型
// ============================================================================

/// Provider API 类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ApiProviderType {
    Openai,
    OpenaiResponse,
    /// Codex CLI 协议（使用 /responses 端点）
    Codex,
    Anthropic,
    /// Anthropic 兼容格式（支持 system 数组格式等变体）
    AnthropicCompatible,
    Gemini,
    AzureOpenai,
    Vertexai,
    AwsBedrock,
    Ollama,
    Fal,
    NewApi,
    Gateway,
}

/// API Key Provider 声明的 Prompt Cache 模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiProviderPromptCacheMode {
    Automatic,
    ExplicitOnly,
}

/// Provider 协议族
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderProtocolFamily {
    OpenAiCompatible,
    Anthropic,
    Gemini,
    AzureOpenai,
    Vertexai,
    AwsBedrock,
    Ollama,
    Codex,
}

/// Provider 运行时规范
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderRuntimeSpec {
    pub protocol_family: ProviderProtocolFamily,
    pub default_api_host: &'static str,
    pub auth_header: &'static str,
    pub auth_prefix: Option<&'static str>,
    pub extra_headers: &'static [(&'static str, &'static str)],
    pub runtime_provider_name: &'static str,
    pub supports_websockets: bool,
}

const NO_EXTRA_HEADERS: [(&str, &str); 0] = [];
const ANTHROPIC_EXTRA_HEADERS: [(&str, &str); 1] = [("anthropic-version", "2023-06-01")];

impl ApiProviderType {
    /// 统一的 Provider 运行时规范
    pub const fn runtime_spec(&self) -> ProviderRuntimeSpec {
        match self {
            ApiProviderType::Anthropic | ApiProviderType::AnthropicCompatible => {
                ProviderRuntimeSpec {
                    protocol_family: ProviderProtocolFamily::Anthropic,
                    default_api_host: "https://api.anthropic.com",
                    auth_header: "x-api-key",
                    auth_prefix: None,
                    extra_headers: &ANTHROPIC_EXTRA_HEADERS,
                    runtime_provider_name: "anthropic",
                    supports_websockets: false,
                }
            }
            ApiProviderType::Gemini => ProviderRuntimeSpec {
                protocol_family: ProviderProtocolFamily::Gemini,
                default_api_host: "https://generativelanguage.googleapis.com",
                auth_header: "x-goog-api-key",
                auth_prefix: None,
                extra_headers: &NO_EXTRA_HEADERS,
                runtime_provider_name: "google",
                supports_websockets: false,
            },
            ApiProviderType::AzureOpenai => ProviderRuntimeSpec {
                protocol_family: ProviderProtocolFamily::AzureOpenai,
                default_api_host: "https://api.openai.com",
                auth_header: "api-key",
                auth_prefix: None,
                extra_headers: &NO_EXTRA_HEADERS,
                runtime_provider_name: "azure",
                supports_websockets: false,
            },
            ApiProviderType::Vertexai => ProviderRuntimeSpec {
                protocol_family: ProviderProtocolFamily::Vertexai,
                default_api_host: "https://api.openai.com",
                auth_header: "Authorization",
                auth_prefix: Some("Bearer"),
                extra_headers: &NO_EXTRA_HEADERS,
                runtime_provider_name: "gcpvertexai",
                supports_websockets: false,
            },
            ApiProviderType::AwsBedrock => ProviderRuntimeSpec {
                protocol_family: ProviderProtocolFamily::AwsBedrock,
                default_api_host: "https://api.openai.com",
                auth_header: "Authorization",
                auth_prefix: Some("Bearer"),
                extra_headers: &NO_EXTRA_HEADERS,
                runtime_provider_name: "bedrock",
                supports_websockets: false,
            },
            ApiProviderType::Ollama => ProviderRuntimeSpec {
                protocol_family: ProviderProtocolFamily::Ollama,
                default_api_host: "http://127.0.0.1:11434",
                auth_header: "Authorization",
                auth_prefix: Some("Bearer"),
                extra_headers: &NO_EXTRA_HEADERS,
                runtime_provider_name: "ollama",
                supports_websockets: false,
            },
            ApiProviderType::Fal => ProviderRuntimeSpec {
                protocol_family: ProviderProtocolFamily::OpenAiCompatible,
                default_api_host: "https://fal.run",
                auth_header: "Authorization",
                auth_prefix: Some("Key"),
                extra_headers: &NO_EXTRA_HEADERS,
                runtime_provider_name: "fal",
                supports_websockets: false,
            },
            ApiProviderType::Codex => ProviderRuntimeSpec {
                protocol_family: ProviderProtocolFamily::Codex,
                default_api_host: "https://api.openai.com",
                auth_header: "Authorization",
                auth_prefix: Some("Bearer"),
                extra_headers: &NO_EXTRA_HEADERS,
                runtime_provider_name: "codex",
                supports_websockets: true,
            },
            ApiProviderType::Openai | ApiProviderType::OpenaiResponse => ProviderRuntimeSpec {
                protocol_family: ProviderProtocolFamily::OpenAiCompatible,
                default_api_host: "https://api.openai.com",
                auth_header: "Authorization",
                auth_prefix: Some("Bearer"),
                extra_headers: &NO_EXTRA_HEADERS,
                runtime_provider_name: "openai",
                supports_websockets: true,
            },
            ApiProviderType::NewApi | ApiProviderType::Gateway => ProviderRuntimeSpec {
                protocol_family: ProviderProtocolFamily::OpenAiCompatible,
                default_api_host: "https://api.openai.com",
                auth_header: "Authorization",
                auth_prefix: Some("Bearer"),
                extra_headers: &NO_EXTRA_HEADERS,
                runtime_provider_name: "openai",
                supports_websockets: false,
            },
        }
    }

    /// 是否属于 Anthropic 协议族
    pub const fn is_anthropic_protocol(&self) -> bool {
        matches!(
            self.runtime_spec().protocol_family,
            ProviderProtocolFamily::Anthropic
        )
    }

    /// 是否支持自动注入 Anthropic prompt cache 标记。
    ///
    /// `AnthropicCompatible` 仅表示协议形状兼容，不代表上游实现了 Anthropic prompt caching。
    pub const fn supports_anthropic_prompt_cache(&self) -> bool {
        matches!(self, ApiProviderType::Anthropic)
    }

    pub const fn default_prompt_cache_mode(&self) -> Option<ApiProviderPromptCacheMode> {
        match self {
            ApiProviderType::Anthropic => Some(ApiProviderPromptCacheMode::Automatic),
            ApiProviderType::AnthropicCompatible => Some(ApiProviderPromptCacheMode::ExplicitOnly),
            _ => None,
        }
    }
}

pub fn infer_managed_runtime_spec(
    provider_type: ApiProviderType,
    api_host: &str,
) -> ProviderRuntimeSpec {
    let effective_provider_type = infer_managed_provider_type(provider_type, api_host);

    if effective_provider_type == ApiProviderType::AnthropicCompatible
        && is_known_automatic_anthropic_compatible_host(Some(api_host))
    {
        return ProviderRuntimeSpec {
            protocol_family: ProviderProtocolFamily::Anthropic,
            default_api_host: "https://api.anthropic.com",
            auth_header: "Authorization",
            auth_prefix: Some("Bearer"),
            extra_headers: &ANTHROPIC_EXTRA_HEADERS,
            runtime_provider_name: "anthropic",
            supports_websockets: false,
        };
    }

    effective_provider_type.runtime_spec()
}

impl std::fmt::Display for ApiProviderPromptCacheMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiProviderPromptCacheMode::Automatic => write!(f, "automatic"),
            ApiProviderPromptCacheMode::ExplicitOnly => write!(f, "explicit_only"),
        }
    }
}

impl std::str::FromStr for ApiProviderPromptCacheMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "automatic" => Ok(Self::Automatic),
            "explicit_only" | "explicit-only" => Ok(Self::ExplicitOnly),
            _ => Err(format!("Unknown prompt cache mode: {s}")),
        }
    }
}

impl std::fmt::Display for ApiProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiProviderType::Openai => write!(f, "openai"),
            ApiProviderType::OpenaiResponse => write!(f, "openai-response"),
            ApiProviderType::Codex => write!(f, "codex"),
            ApiProviderType::Anthropic => write!(f, "anthropic"),
            ApiProviderType::AnthropicCompatible => write!(f, "anthropic-compatible"),
            ApiProviderType::Gemini => write!(f, "gemini"),
            ApiProviderType::AzureOpenai => write!(f, "azure-openai"),
            ApiProviderType::Vertexai => write!(f, "vertexai"),
            ApiProviderType::AwsBedrock => write!(f, "aws-bedrock"),
            ApiProviderType::Ollama => write!(f, "ollama"),
            ApiProviderType::Fal => write!(f, "fal"),
            ApiProviderType::NewApi => write!(f, "new-api"),
            ApiProviderType::Gateway => write!(f, "gateway"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        infer_managed_prompt_cache_mode, infer_managed_provider_type, infer_managed_runtime_spec,
        ApiKeyProvider, ApiProviderPromptCacheMode, ApiProviderType, ProviderGroup,
        ProviderProtocolFamily,
    };
    use chrono::Utc;

    #[test]
    fn test_runtime_spec_anthropic_compatible() {
        let spec = ApiProviderType::AnthropicCompatible.runtime_spec();
        assert_eq!(spec.protocol_family, ProviderProtocolFamily::Anthropic);
        assert_eq!(spec.auth_header, "x-api-key");
        assert_eq!(spec.auth_prefix, None);
        assert_eq!(spec.default_api_host, "https://api.anthropic.com");
        assert_eq!(spec.extra_headers[0], ("anthropic-version", "2023-06-01"));
    }

    #[test]
    fn test_runtime_spec_openai_defaults() {
        let spec = ApiProviderType::Openai.runtime_spec();
        assert_eq!(
            spec.protocol_family,
            ProviderProtocolFamily::OpenAiCompatible
        );
        assert_eq!(spec.auth_header, "Authorization");
        assert_eq!(spec.auth_prefix, Some("Bearer"));
        assert_eq!(spec.default_api_host, "https://api.openai.com");
    }

    #[test]
    fn test_supports_anthropic_prompt_cache() {
        assert!(ApiProviderType::Anthropic.supports_anthropic_prompt_cache());
        assert!(!ApiProviderType::AnthropicCompatible.supports_anthropic_prompt_cache());
        assert!(!ApiProviderType::Openai.supports_anthropic_prompt_cache());
    }

    #[test]
    fn test_default_prompt_cache_mode() {
        assert_eq!(
            ApiProviderType::Anthropic.default_prompt_cache_mode(),
            Some(ApiProviderPromptCacheMode::Automatic)
        );
        assert_eq!(
            ApiProviderType::AnthropicCompatible.default_prompt_cache_mode(),
            Some(ApiProviderPromptCacheMode::ExplicitOnly)
        );
        assert_eq!(ApiProviderType::Openai.default_prompt_cache_mode(), None);
    }

    #[test]
    fn test_known_official_anthropic_compatible_hosts_default_to_automatic() {
        let hosts = [
            "https://open.bigmodel.cn/api/anthropic",
            "https://api.z.ai/api/anthropic",
            "https://api.moonshot.cn/anthropic",
            "https://api.moonshot.ai/anthropic",
            "https://api.kimi.com/coding/",
            "https://api.minimaxi.com/anthropic",
            "https://api.minimax.io/anthropic",
            "https://coding.dashscope.aliyuncs.com/apps/anthropic",
            "https://coding-intl.dashscope.aliyuncs.com/apps/anthropic",
            "https://token-plan-cn.xiaomimimo.com/anthropic",
        ];

        for host in hosts {
            assert_eq!(
                infer_managed_prompt_cache_mode(ApiProviderType::AnthropicCompatible, host),
                Some(ApiProviderPromptCacheMode::Automatic),
                "expected host to resolve automatic prompt cache: {host}"
            );
        }
    }

    #[test]
    fn test_effective_prompt_cache_mode_prefers_known_host_inference() {
        let provider = ApiKeyProvider {
            id: "custom-provider".to_string(),
            name: "Official Anthropic-Compatible".to_string(),
            provider_type: ApiProviderType::Openai,
            api_host: "https://api.minimaxi.com/anthropic".to_string(),
            is_system: false,
            group: ProviderGroup::Custom,
            enabled: true,
            sort_order: 9999,
            api_version: None,
            project: None,
            location: None,
            region: None,
            custom_models: Vec::new(),
            prompt_cache_mode: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        assert_eq!(
            provider.effective_prompt_cache_mode(),
            Some(ApiProviderPromptCacheMode::Automatic)
        );
        assert_eq!(
            provider.effective_provider_type(),
            ApiProviderType::AnthropicCompatible
        );
    }

    #[test]
    fn test_known_official_anthropic_compatible_hosts_should_normalize_provider_type() {
        for provider_type in [
            ApiProviderType::Openai,
            ApiProviderType::OpenaiResponse,
            ApiProviderType::Anthropic,
            ApiProviderType::Gateway,
        ] {
            assert_eq!(
                infer_managed_provider_type(provider_type, "https://api.minimaxi.com/anthropic"),
                ApiProviderType::AnthropicCompatible
            );
        }

        assert_eq!(
            infer_managed_provider_type(
                ApiProviderType::Gemini,
                "https://api.minimaxi.com/anthropic"
            ),
            ApiProviderType::Gemini
        );
    }

    #[test]
    fn test_known_official_anthropic_compatible_hosts_should_use_authorization_header() {
        let spec = infer_managed_runtime_spec(
            ApiProviderType::Anthropic,
            "https://api.minimaxi.com/anthropic",
        );

        assert_eq!(spec.protocol_family, ProviderProtocolFamily::Anthropic);
        assert_eq!(spec.auth_header, "Authorization");
        assert_eq!(spec.auth_prefix, Some("Bearer"));
        assert_eq!(spec.extra_headers[0], ("anthropic-version", "2023-06-01"));

        let xiaomi_sgp_spec = infer_managed_runtime_spec(
            ApiProviderType::Openai,
            "https://token-plan-sgp.xiaomimimo.com/anthropic",
        );
        assert_eq!(
            xiaomi_sgp_spec.protocol_family,
            ProviderProtocolFamily::Anthropic
        );
        assert_eq!(xiaomi_sgp_spec.auth_header, "Authorization");
        assert_eq!(xiaomi_sgp_spec.auth_prefix, Some("Bearer"));
    }

    #[test]
    fn test_unknown_anthropic_hosts_should_keep_x_api_key_header() {
        let spec =
            infer_managed_runtime_spec(ApiProviderType::Anthropic, "https://api.anthropic.com");

        assert_eq!(spec.protocol_family, ProviderProtocolFamily::Anthropic);
        assert_eq!(spec.auth_header, "x-api-key");
        assert_eq!(spec.auth_prefix, None);
    }

    #[test]
    fn test_runtime_spec_contract_matrix() {
        let cases = [
            (
                ApiProviderType::Openai,
                ProviderProtocolFamily::OpenAiCompatible,
                "https://api.openai.com",
                "Authorization",
                Some("Bearer"),
                "openai",
            ),
            (
                ApiProviderType::OpenaiResponse,
                ProviderProtocolFamily::OpenAiCompatible,
                "https://api.openai.com",
                "Authorization",
                Some("Bearer"),
                "openai",
            ),
            (
                ApiProviderType::Codex,
                ProviderProtocolFamily::Codex,
                "https://api.openai.com",
                "Authorization",
                Some("Bearer"),
                "codex",
            ),
            (
                ApiProviderType::Anthropic,
                ProviderProtocolFamily::Anthropic,
                "https://api.anthropic.com",
                "x-api-key",
                None,
                "anthropic",
            ),
            (
                ApiProviderType::AnthropicCompatible,
                ProviderProtocolFamily::Anthropic,
                "https://api.anthropic.com",
                "x-api-key",
                None,
                "anthropic",
            ),
            (
                ApiProviderType::Gemini,
                ProviderProtocolFamily::Gemini,
                "https://generativelanguage.googleapis.com",
                "x-goog-api-key",
                None,
                "google",
            ),
            (
                ApiProviderType::AzureOpenai,
                ProviderProtocolFamily::AzureOpenai,
                "https://api.openai.com",
                "api-key",
                None,
                "azure",
            ),
            (
                ApiProviderType::Vertexai,
                ProviderProtocolFamily::Vertexai,
                "https://api.openai.com",
                "Authorization",
                Some("Bearer"),
                "gcpvertexai",
            ),
            (
                ApiProviderType::AwsBedrock,
                ProviderProtocolFamily::AwsBedrock,
                "https://api.openai.com",
                "Authorization",
                Some("Bearer"),
                "bedrock",
            ),
            (
                ApiProviderType::Ollama,
                ProviderProtocolFamily::Ollama,
                "http://127.0.0.1:11434",
                "Authorization",
                Some("Bearer"),
                "ollama",
            ),
            (
                ApiProviderType::Fal,
                ProviderProtocolFamily::OpenAiCompatible,
                "https://fal.run",
                "Authorization",
                Some("Key"),
                "fal",
            ),
            (
                ApiProviderType::NewApi,
                ProviderProtocolFamily::OpenAiCompatible,
                "https://api.openai.com",
                "Authorization",
                Some("Bearer"),
                "openai",
            ),
            (
                ApiProviderType::Gateway,
                ProviderProtocolFamily::OpenAiCompatible,
                "https://api.openai.com",
                "Authorization",
                Some("Bearer"),
                "openai",
            ),
        ];

        for (
            provider_type,
            expected_family,
            expected_host,
            expected_auth_header,
            expected_auth_prefix,
            expected_runtime_provider,
        ) in cases
        {
            let spec = provider_type.runtime_spec();
            assert_eq!(
                spec.protocol_family, expected_family,
                "provider_type={provider_type:?}"
            );
            assert_eq!(
                spec.default_api_host, expected_host,
                "provider_type={provider_type:?}"
            );
            assert_eq!(
                spec.auth_header, expected_auth_header,
                "provider_type={provider_type:?}"
            );
            assert_eq!(
                spec.auth_prefix, expected_auth_prefix,
                "provider_type={provider_type:?}"
            );
            assert_eq!(
                spec.runtime_provider_name, expected_runtime_provider,
                "provider_type={provider_type:?}"
            );
        }

        let anthropic_spec = ApiProviderType::Anthropic.runtime_spec();
        assert_eq!(
            anthropic_spec.extra_headers,
            &[("anthropic-version", "2023-06-01")]
        );
        assert!(ApiProviderType::Openai.runtime_spec().supports_websockets);
        assert!(
            ApiProviderType::OpenaiResponse
                .runtime_spec()
                .supports_websockets
        );
        assert!(ApiProviderType::Codex.runtime_spec().supports_websockets);
        assert!(!ApiProviderType::Gateway.runtime_spec().supports_websockets);
        assert!(!ApiProviderType::NewApi.runtime_spec().supports_websockets);

        let anthropic_compatible_spec = ApiProviderType::AnthropicCompatible.runtime_spec();
        assert_eq!(
            anthropic_compatible_spec.extra_headers,
            &[("anthropic-version", "2023-06-01")]
        );

        let openai_spec = ApiProviderType::Openai.runtime_spec();
        assert!(openai_spec.extra_headers.is_empty());
    }
}

impl std::str::FromStr for ApiProviderType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(ApiProviderType::Openai),
            "openai-response" => Ok(ApiProviderType::OpenaiResponse),
            "codex" => Ok(ApiProviderType::Codex),
            "anthropic" => Ok(ApiProviderType::Anthropic),
            "anthropic-compatible" => Ok(ApiProviderType::AnthropicCompatible),
            "gemini" => Ok(ApiProviderType::Gemini),
            "azure-openai" => Ok(ApiProviderType::AzureOpenai),
            "vertexai" => Ok(ApiProviderType::Vertexai),
            "aws-bedrock" => Ok(ApiProviderType::AwsBedrock),
            "ollama" => Ok(ApiProviderType::Ollama),
            "fal" => Ok(ApiProviderType::Fal),
            "new-api" => Ok(ApiProviderType::NewApi),
            "gateway" => Ok(ApiProviderType::Gateway),
            _ => Err(format!("Invalid provider type: {s}")),
        }
    }
}

/// Provider 分组类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderGroup {
    Mainstream,
    Chinese,
    Cloud,
    Aggregator,
    Local,
    Specialized,
    Custom,
}

impl std::fmt::Display for ProviderGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderGroup::Mainstream => write!(f, "mainstream"),
            ProviderGroup::Chinese => write!(f, "chinese"),
            ProviderGroup::Cloud => write!(f, "cloud"),
            ProviderGroup::Aggregator => write!(f, "aggregator"),
            ProviderGroup::Local => write!(f, "local"),
            ProviderGroup::Specialized => write!(f, "specialized"),
            ProviderGroup::Custom => write!(f, "custom"),
        }
    }
}

impl std::str::FromStr for ProviderGroup {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "mainstream" => Ok(ProviderGroup::Mainstream),
            "chinese" => Ok(ProviderGroup::Chinese),
            "cloud" => Ok(ProviderGroup::Cloud),
            "aggregator" => Ok(ProviderGroup::Aggregator),
            "local" => Ok(ProviderGroup::Local),
            "specialized" => Ok(ProviderGroup::Specialized),
            "custom" => Ok(ProviderGroup::Custom),
            _ => Err(format!("Invalid provider group: {s}")),
        }
    }
}

/// API Key Provider 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyProvider {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub provider_type: ApiProviderType,
    pub api_host: String,
    pub is_system: bool,
    pub group: ProviderGroup,
    pub enabled: bool,
    pub sort_order: i32,
    pub api_version: Option<String>,
    pub project: Option<String>,
    pub location: Option<String>,
    pub region: Option<String>,
    /// 自定义模型列表（JSON 数组格式存储）
    /// 用于不支持 /models 接口的 Provider（如智谱）
    #[serde(default)]
    pub custom_models: Vec<String>,
    /// Provider 显式声明的 Prompt Cache 模式（仅在需要覆盖类型默认值时设置）
    #[serde(default)]
    pub prompt_cache_mode: Option<ApiProviderPromptCacheMode>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

fn infer_managed_prompt_cache_mode(
    provider_type: ApiProviderType,
    api_host: &str,
) -> Option<ApiProviderPromptCacheMode> {
    if provider_type == ApiProviderType::AnthropicCompatible
        && is_known_automatic_anthropic_compatible_host(Some(api_host))
    {
        return Some(ApiProviderPromptCacheMode::Automatic);
    }

    None
}

pub fn infer_managed_provider_type(
    provider_type: ApiProviderType,
    api_host: &str,
) -> ApiProviderType {
    if is_known_automatic_anthropic_compatible_host(Some(api_host)) {
        match provider_type {
            ApiProviderType::Anthropic
            | ApiProviderType::AnthropicCompatible
            | ApiProviderType::Openai
            | ApiProviderType::OpenaiResponse
            | ApiProviderType::NewApi
            | ApiProviderType::Gateway => return ApiProviderType::AnthropicCompatible,
            _ => {}
        }
    }

    provider_type
}

impl ApiKeyProvider {
    pub fn effective_provider_type(&self) -> ApiProviderType {
        infer_managed_provider_type(self.provider_type, &self.api_host)
    }

    pub fn effective_runtime_spec(&self) -> ProviderRuntimeSpec {
        infer_managed_runtime_spec(self.provider_type, &self.api_host)
    }

    pub fn effective_prompt_cache_mode(&self) -> Option<ApiProviderPromptCacheMode> {
        let effective_provider_type = self.effective_provider_type();

        if let Some(managed_mode) =
            infer_managed_prompt_cache_mode(effective_provider_type, &self.api_host)
        {
            return Some(managed_mode);
        }

        match effective_provider_type {
            ApiProviderType::AnthropicCompatible => self
                .prompt_cache_mode
                .or_else(|| effective_provider_type.default_prompt_cache_mode()),
            _ => effective_provider_type.default_prompt_cache_mode(),
        }
    }

    pub fn supports_automatic_prompt_cache(&self) -> bool {
        matches!(
            self.effective_prompt_cache_mode(),
            Some(ApiProviderPromptCacheMode::Automatic)
        )
    }
}

/// API Key 条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyEntry {
    pub id: String,
    pub provider_id: String,
    /// 加密后的 API Key
    pub api_key_encrypted: String,
    pub alias: Option<String>,
    pub enabled: bool,
    pub usage_count: i64,
    pub error_count: i64,
    pub last_used_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

/// Provider 完整数据（包含 API Keys）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderWithKeys {
    #[serde(flatten)]
    pub provider: ApiKeyProvider,
    pub api_keys: Vec<ApiKeyEntry>,
}

// ============================================================================
// DAO 实现
// ============================================================================

pub struct ApiKeyProviderDao;

fn parse_provider_type_column(
    value: &str,
    column_index: usize,
) -> Result<ApiProviderType, rusqlite::Error> {
    value.parse().map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(
            column_index,
            rusqlite::types::Type::Text,
            Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, error)),
        )
    })
}

impl ApiKeyProviderDao {
    // ==================== Provider 操作 ====================

    /// 获取所有 Provider
    pub fn get_all_providers(conn: &Connection) -> Result<Vec<ApiKeyProvider>, rusqlite::Error> {
        let mut stmt = conn.prepare(
            "SELECT id, name, type, api_host, is_system, group_name, enabled, sort_order,
                    api_version, project, location, region, custom_models, prompt_cache_mode,
                    created_at, updated_at
             FROM api_key_providers
             ORDER BY sort_order ASC, created_at ASC",
        )?;

        let rows = stmt.query_map([], Self::row_to_provider)?;
        let mut providers = Vec::new();
        for provider in rows.flatten() {
            providers.push(provider);
        }
        Ok(providers)
    }

    /// 根据 ID 获取 Provider
    pub fn get_provider_by_id(
        conn: &Connection,
        id: &str,
    ) -> Result<Option<ApiKeyProvider>, rusqlite::Error> {
        let mut stmt = conn.prepare(
            "SELECT id, name, type, api_host, is_system, group_name, enabled, sort_order,
                    api_version, project, location, region, custom_models, prompt_cache_mode,
                    created_at, updated_at
             FROM api_key_providers
             WHERE id = ?1",
        )?;

        let mut rows = stmt.query([id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(Self::row_to_provider(row)?))
        } else {
            Ok(None)
        }
    }

    /// 根据分组获取 Provider
    pub fn get_providers_by_group(
        conn: &Connection,
        group: ProviderGroup,
    ) -> Result<Vec<ApiKeyProvider>, rusqlite::Error> {
        let mut stmt = conn.prepare(
            "SELECT id, name, type, api_host, is_system, group_name, enabled, sort_order,
                    api_version, project, location, region, custom_models, prompt_cache_mode,
                    created_at, updated_at
             FROM api_key_providers
             WHERE group_name = ?1
             ORDER BY sort_order ASC, created_at ASC",
        )?;

        let rows = stmt.query_map([group.to_string()], Self::row_to_provider)?;
        let mut providers = Vec::new();
        for provider in rows.flatten() {
            providers.push(provider);
        }
        Ok(providers)
    }

    /// 插入新 Provider
    pub fn insert_provider(
        conn: &Connection,
        provider: &ApiKeyProvider,
    ) -> Result<(), rusqlite::Error> {
        let custom_models_json = if provider.custom_models.is_empty() {
            None
        } else {
            Some(serde_json::to_string(&provider.custom_models).unwrap_or_default())
        };
        let prompt_cache_mode = provider.prompt_cache_mode.map(|value| value.to_string());

        conn.execute(
            "INSERT INTO api_key_providers
             (id, name, type, api_host, is_system, group_name, enabled, sort_order,
              api_version, project, location, region, custom_models, prompt_cache_mode, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
            params![
                provider.id,
                provider.name,
                provider.provider_type.to_string(),
                provider.api_host,
                provider.is_system,
                provider.group.to_string(),
                provider.enabled,
                provider.sort_order,
                provider.api_version,
                provider.project,
                provider.location,
                provider.region,
                custom_models_json,
                prompt_cache_mode,
                provider.created_at.to_rfc3339(),
                provider.updated_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    /// 更新 Provider
    pub fn update_provider(
        conn: &Connection,
        provider: &ApiKeyProvider,
    ) -> Result<(), rusqlite::Error> {
        let custom_models_json = if provider.custom_models.is_empty() {
            None
        } else {
            Some(serde_json::to_string(&provider.custom_models).unwrap_or_default())
        };
        let prompt_cache_mode = provider.prompt_cache_mode.map(|value| value.to_string());

        conn.execute(
            "UPDATE api_key_providers SET
             name = ?2, type = ?3, api_host = ?4, is_system = ?5, group_name = ?6,
             enabled = ?7, sort_order = ?8, api_version = ?9, project = ?10,
             location = ?11, region = ?12, custom_models = ?13, prompt_cache_mode = ?14, updated_at = ?15
             WHERE id = ?1",
            params![
                provider.id,
                provider.name,
                provider.provider_type.to_string(),
                provider.api_host,
                provider.is_system,
                provider.group.to_string(),
                provider.enabled,
                provider.sort_order,
                provider.api_version,
                provider.project,
                provider.location,
                provider.region,
                custom_models_json,
                prompt_cache_mode,
                provider.updated_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    /// 删除 Provider（仅限自定义 Provider）
    pub fn delete_provider(conn: &Connection, id: &str) -> Result<bool, rusqlite::Error> {
        // 先检查是否为系统 Provider
        let is_system: bool = conn.query_row(
            "SELECT is_system FROM api_key_providers WHERE id = ?1",
            [id],
            |row| row.get(0),
        )?;

        if is_system {
            return Ok(false); // 不允许删除系统 Provider
        }

        let affected = conn.execute("DELETE FROM api_key_providers WHERE id = ?1", [id])?;
        Ok(affected > 0)
    }

    /// 从数据库行转换为 ApiKeyProvider
    fn row_to_provider(row: &rusqlite::Row) -> Result<ApiKeyProvider, rusqlite::Error> {
        let id: String = row.get(0)?;
        let name: String = row.get(1)?;
        let type_str: String = row.get(2)?;
        let api_host: String = row.get(3)?;
        let is_system: bool = row.get(4)?;
        let group_str: String = row.get(5)?;
        let enabled: bool = row.get(6)?;
        let sort_order: i32 = row.get(7)?;
        let api_version: Option<String> = row.get(8)?;
        let project: Option<String> = row.get(9)?;
        let location: Option<String> = row.get(10)?;
        let region: Option<String> = row.get(11)?;
        let custom_models_json: Option<String> = row.get(12)?;
        let prompt_cache_mode_str: Option<String> = row.get(13)?;
        let created_at_str: String = row.get(14)?;
        let updated_at_str: String = row.get(15)?;

        let provider_type = parse_provider_type_column(&type_str, 2)?;
        let group: ProviderGroup = group_str.parse().unwrap_or(ProviderGroup::Custom);

        let created_at = DateTime::parse_from_rfc3339(&created_at_str)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());
        let updated_at = DateTime::parse_from_rfc3339(&updated_at_str)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        // 解析自定义模型列表
        let custom_models: Vec<String> = custom_models_json
            .and_then(|json| serde_json::from_str(&json).ok())
            .unwrap_or_default();
        let prompt_cache_mode = prompt_cache_mode_str
            .and_then(|value| value.parse::<ApiProviderPromptCacheMode>().ok());

        Ok(ApiKeyProvider {
            id,
            name,
            provider_type,
            api_host,
            is_system,
            group,
            enabled,
            sort_order,
            api_version,
            project,
            location,
            region,
            custom_models,
            prompt_cache_mode,
            created_at,
            updated_at,
        })
    }

    // ==================== API Key 操作 ====================

    /// 获取 Provider 的所有 API Keys
    pub fn get_api_keys_by_provider(
        conn: &Connection,
        provider_id: &str,
    ) -> Result<Vec<ApiKeyEntry>, rusqlite::Error> {
        let mut stmt = conn.prepare(
            "SELECT id, provider_id, api_key_encrypted, alias, enabled,
                    usage_count, error_count, last_used_at, created_at
             FROM api_keys
             WHERE provider_id = ?1
             ORDER BY created_at ASC",
        )?;

        let rows = stmt.query_map([provider_id], Self::row_to_api_key)?;
        let mut keys = Vec::new();
        for key in rows.flatten() {
            keys.push(key);
        }
        Ok(keys)
    }

    /// 获取所有启用的 API Keys
    pub fn get_enabled_api_keys_by_provider(
        conn: &Connection,
        provider_id: &str,
    ) -> Result<Vec<ApiKeyEntry>, rusqlite::Error> {
        let mut stmt = conn.prepare(
            "SELECT id, provider_id, api_key_encrypted, alias, enabled,
                    usage_count, error_count, last_used_at, created_at
             FROM api_keys
             WHERE provider_id = ?1 AND enabled = 1
             ORDER BY created_at ASC",
        )?;

        let rows = stmt.query_map([provider_id], Self::row_to_api_key)?;
        let mut keys = Vec::new();
        for key in rows.flatten() {
            keys.push(key);
        }
        Ok(keys)
    }

    /// 获取指定类型的所有启用的 API Keys（包括自定义 Provider）
    /// 返回 (ApiKeyEntry, ApiKeyProvider) 元组列表
    pub fn get_enabled_api_keys_by_type(
        conn: &Connection,
        provider_type: ApiProviderType,
    ) -> Result<Vec<(ApiKeyEntry, ApiKeyProvider)>, rusqlite::Error> {
        let type_str = provider_type.to_string();
        let mut stmt = conn.prepare(
            "SELECT k.id, k.provider_id, k.api_key_encrypted, k.alias, k.enabled,
                    k.usage_count, k.error_count, k.last_used_at, k.created_at,
                    p.id, p.name, p.type, p.api_host, p.is_system, p.group_name, p.enabled,
                    p.sort_order, p.api_version, p.project, p.location, p.region,
                    p.custom_models, p.prompt_cache_mode, p.created_at, p.updated_at
             FROM api_keys k
             JOIN api_key_providers p ON k.provider_id = p.id
             WHERE p.type = ?1 AND k.enabled = 1 AND p.enabled = 1
             ORDER BY p.sort_order ASC, k.created_at ASC",
        )?;

        let rows = stmt.query_map([type_str], |row| {
            // 解析 API Key
            let last_used_at_str: Option<String> = row.get(7)?;
            let created_at_str: String = row.get(8)?;
            let last_used_at = last_used_at_str.and_then(|s| {
                DateTime::parse_from_rfc3339(&s)
                    .ok()
                    .map(|dt| dt.with_timezone(&Utc))
            });
            let key_created_at = DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            let key = ApiKeyEntry {
                id: row.get(0)?,
                provider_id: row.get(1)?,
                api_key_encrypted: row.get(2)?,
                alias: row.get(3)?,
                enabled: row.get(4)?,
                usage_count: row.get(5)?,
                error_count: row.get(6)?,
                last_used_at,
                created_at: key_created_at,
            };

            // 解析 Provider
            let custom_models_json: Option<String> = row.get(21)?;
            let prompt_cache_mode_str: Option<String> = row.get(22)?;
            let provider_created_at_str: String = row.get(23)?;
            let provider_updated_at_str: String = row.get(24)?;
            let provider_created_at = DateTime::parse_from_rfc3339(&provider_created_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());
            let provider_updated_at = DateTime::parse_from_rfc3339(&provider_updated_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            // 解析自定义模型列表
            let custom_models: Vec<String> = custom_models_json
                .and_then(|json| serde_json::from_str(&json).ok())
                .unwrap_or_default();
            let prompt_cache_mode = prompt_cache_mode_str
                .and_then(|value| value.parse::<ApiProviderPromptCacheMode>().ok());

            let provider = ApiKeyProvider {
                id: row.get(9)?,
                name: row.get(10)?,
                provider_type: parse_provider_type_column(&row.get::<_, String>(11)?, 11)?,
                api_host: row.get(12)?,
                is_system: row.get(13)?,
                group: row
                    .get::<_, String>(14)?
                    .parse()
                    .unwrap_or(ProviderGroup::Custom),
                enabled: row.get(15)?,
                sort_order: row.get(16)?,
                api_version: row.get(17)?,
                project: row.get(18)?,
                location: row.get(19)?,
                region: row.get(20)?,
                custom_models,
                prompt_cache_mode,
                created_at: provider_created_at,
                updated_at: provider_updated_at,
            };

            Ok((key, provider))
        })?;

        let mut result = Vec::new();
        for item in rows.flatten() {
            result.push(item);
        }
        Ok(result)
    }

    /// 根据 ID 获取 API Key
    pub fn get_api_key_by_id(
        conn: &Connection,
        id: &str,
    ) -> Result<Option<ApiKeyEntry>, rusqlite::Error> {
        let mut stmt = conn.prepare(
            "SELECT id, provider_id, api_key_encrypted, alias, enabled,
                    usage_count, error_count, last_used_at, created_at
             FROM api_keys
             WHERE id = ?1",
        )?;

        let mut rows = stmt.query([id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(Self::row_to_api_key(row)?))
        } else {
            Ok(None)
        }
    }

    /// 插入新 API Key
    pub fn insert_api_key(conn: &Connection, key: &ApiKeyEntry) -> Result<(), rusqlite::Error> {
        tracing::info!(
            "[DAO] insert_api_key: id={}, provider_id={}",
            key.id,
            key.provider_id
        );

        conn.execute(
            "INSERT INTO api_keys
             (id, provider_id, api_key_encrypted, alias, enabled,
              usage_count, error_count, last_used_at, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                key.id,
                key.provider_id,
                key.api_key_encrypted,
                key.alias,
                key.enabled,
                key.usage_count,
                key.error_count,
                key.last_used_at.map(|t| t.to_rfc3339()),
                key.created_at.to_rfc3339(),
            ],
        )?;

        tracing::info!("[DAO] insert_api_key: 插入成功");
        Ok(())
    }

    /// 更新 API Key
    pub fn update_api_key(conn: &Connection, key: &ApiKeyEntry) -> Result<(), rusqlite::Error> {
        conn.execute(
            "UPDATE api_keys SET
             alias = ?2, enabled = ?3, usage_count = ?4, error_count = ?5, last_used_at = ?6
             WHERE id = ?1",
            params![
                key.id,
                key.alias,
                key.enabled,
                key.usage_count,
                key.error_count,
                key.last_used_at.map(|t| t.to_rfc3339()),
            ],
        )?;
        Ok(())
    }

    pub fn update_api_key_encrypted(
        conn: &Connection,
        id: &str,
        api_key_encrypted: &str,
    ) -> Result<(), rusqlite::Error> {
        conn.execute(
            "UPDATE api_keys SET api_key_encrypted = ?2 WHERE id = ?1",
            params![id, api_key_encrypted],
        )?;
        Ok(())
    }

    /// 删除 API Key
    pub fn delete_api_key(conn: &Connection, id: &str) -> Result<bool, rusqlite::Error> {
        let affected = conn.execute("DELETE FROM api_keys WHERE id = ?1", [id])?;
        Ok(affected > 0)
    }

    /// 更新 API Key 使用统计
    pub fn update_api_key_usage(
        conn: &Connection,
        id: &str,
        usage_count: i64,
        last_used_at: DateTime<Utc>,
    ) -> Result<(), rusqlite::Error> {
        conn.execute(
            "UPDATE api_keys SET usage_count = ?2, last_used_at = ?3 WHERE id = ?1",
            params![id, usage_count, last_used_at.to_rfc3339()],
        )?;
        Ok(())
    }

    /// 增加 API Key 错误计数
    pub fn increment_api_key_error(conn: &Connection, id: &str) -> Result<(), rusqlite::Error> {
        conn.execute(
            "UPDATE api_keys SET error_count = error_count + 1 WHERE id = ?1",
            [id],
        )?;
        Ok(())
    }

    /// 从数据库行转换为 ApiKeyEntry
    fn row_to_api_key(row: &rusqlite::Row) -> Result<ApiKeyEntry, rusqlite::Error> {
        let id: String = row.get(0)?;
        let provider_id: String = row.get(1)?;
        let api_key_encrypted: String = row.get(2)?;
        let alias: Option<String> = row.get(3)?;
        let enabled: bool = row.get(4)?;
        let usage_count: i64 = row.get(5)?;
        let error_count: i64 = row.get(6)?;
        let last_used_at_str: Option<String> = row.get(7)?;
        let created_at_str: String = row.get(8)?;

        let last_used_at = last_used_at_str.and_then(|s| {
            DateTime::parse_from_rfc3339(&s)
                .ok()
                .map(|dt| dt.with_timezone(&Utc))
        });
        let created_at = DateTime::parse_from_rfc3339(&created_at_str)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        Ok(ApiKeyEntry {
            id,
            provider_id,
            api_key_encrypted,
            alias,
            enabled,
            usage_count,
            error_count,
            last_used_at,
            created_at,
        })
    }

    // ==================== UI 状态操作 ====================

    /// 获取 UI 状态
    pub fn get_ui_state(conn: &Connection, key: &str) -> Result<Option<String>, rusqlite::Error> {
        let mut stmt = conn.prepare("SELECT value FROM provider_ui_state WHERE key = ?1")?;
        let mut rows = stmt.query([key])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row.get(0)?))
        } else {
            Ok(None)
        }
    }

    /// 设置 UI 状态
    pub fn set_ui_state(conn: &Connection, key: &str, value: &str) -> Result<(), rusqlite::Error> {
        conn.execute(
            "INSERT OR REPLACE INTO provider_ui_state (key, value) VALUES (?1, ?2)",
            params![key, value],
        )?;
        Ok(())
    }

    /// 删除 UI 状态
    pub fn delete_ui_state(conn: &Connection, key: &str) -> Result<bool, rusqlite::Error> {
        let affected = conn.execute("DELETE FROM provider_ui_state WHERE key = ?1", [key])?;
        Ok(affected > 0)
    }

    // ==================== 复合查询 ====================

    /// 获取所有 Provider 及其 API Keys
    pub fn get_all_providers_with_keys(
        conn: &Connection,
    ) -> Result<Vec<ProviderWithKeys>, rusqlite::Error> {
        let providers = Self::get_all_providers(conn)?;
        tracing::debug!(
            "[DAO] get_all_providers_with_keys: 获取到 {} 个 Provider",
            providers.len()
        );

        let mut result = Vec::new();

        for provider in providers {
            let api_keys = Self::get_api_keys_by_provider(conn, &provider.id)?;
            tracing::trace!(
                "[DAO] Provider {} ({}): {} 个 API Key",
                provider.id,
                provider.name,
                api_keys.len()
            );
            result.push(ProviderWithKeys { provider, api_keys });
        }

        Ok(result)
    }

    /// 获取启用的 Provider 及其启用的 API Keys
    pub fn get_enabled_providers_with_keys(
        conn: &Connection,
    ) -> Result<Vec<ProviderWithKeys>, rusqlite::Error> {
        let mut stmt = conn.prepare(
            "SELECT id, name, type, api_host, is_system, group_name, enabled, sort_order,
                    api_version, project, location, region, custom_models, created_at, updated_at
             FROM api_key_providers
             WHERE enabled = 1
             ORDER BY sort_order ASC, created_at ASC",
        )?;

        let rows = stmt.query_map([], Self::row_to_provider)?;
        let mut result = Vec::new();

        for provider in rows.flatten() {
            let api_keys = Self::get_enabled_api_keys_by_provider(conn, &provider.id)?;
            if !api_keys.is_empty() {
                result.push(ProviderWithKeys { provider, api_keys });
            }
        }

        Ok(result)
    }

    /// 统计 Provider 的 API Key 数量
    pub fn count_api_keys_by_provider(
        conn: &Connection,
        provider_id: &str,
    ) -> Result<i64, rusqlite::Error> {
        conn.query_row(
            "SELECT COUNT(*) FROM api_keys WHERE provider_id = ?1",
            [provider_id],
            |row| row.get(0),
        )
    }

    /// 批量更新 Provider 排序顺序
    /// **Validates: Requirements 8.4**
    pub fn update_provider_sort_orders(
        conn: &Connection,
        sort_orders: &[(String, i32)],
    ) -> Result<(), rusqlite::Error> {
        let now = chrono::Utc::now().to_rfc3339();
        for (id, sort_order) in sort_orders {
            conn.execute(
                "UPDATE api_key_providers SET sort_order = ?2, updated_at = ?3 WHERE id = ?1",
                params![id, sort_order, now],
            )?;
        }
        Ok(())
    }
}
