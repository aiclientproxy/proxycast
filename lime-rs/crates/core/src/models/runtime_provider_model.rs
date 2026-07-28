//! API Key Provider 运行时 DTO 边界。
//!
//! 旧凭证池运行时模型已退役；current 运行时只通过本模块暴露的 DTO
//! 消费 API Key Provider 必要字段，避免把凭证池心智重新带回业务层。

use super::provider_type::is_custom_provider_id;
use crate::provider_prompt_cache_support::is_known_automatic_anthropic_compatible_host;
use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};
use uuid::Uuid;

/// Current API Key Provider 运行时使用的 Provider 类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeProviderType {
    #[serde(rename = "openai")]
    OpenAI,
    Claude,
    #[serde(rename = "anthropic_compatible")]
    AnthropicCompatible,
    Vertex,
    GeminiApiKey,
    Anthropic,
    AzureOpenai,
    AwsBedrock,
    Ollama,
}

/// API Key Provider 运行时临时凭证 UUID 前缀。
pub const RUNTIME_API_KEY_CREDENTIAL_UUID_PREFIX: &str = "runtime-api-key-";

pub fn runtime_api_key_credential_uuid(api_key_id: &str) -> String {
    format!("{RUNTIME_API_KEY_CREDENTIAL_UUID_PREFIX}{api_key_id}")
}

pub fn runtime_api_key_id_from_credential_uuid(uuid: &str) -> Option<&str> {
    uuid.strip_prefix(RUNTIME_API_KEY_CREDENTIAL_UUID_PREFIX)
        .filter(|value| !value.is_empty())
}

impl RuntimeProviderType {
    /// 是否支持自动注入 Anthropic prompt cache 标记。
    pub const fn supports_anthropic_prompt_cache(&self) -> bool {
        matches!(self, Self::Claude | Self::Anthropic)
    }
}

impl fmt::Display for RuntimeProviderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeProviderType::OpenAI => write!(f, "openai"),
            RuntimeProviderType::Claude => write!(f, "claude"),
            RuntimeProviderType::AnthropicCompatible => write!(f, "anthropic_compatible"),
            RuntimeProviderType::Vertex => write!(f, "vertex"),
            RuntimeProviderType::GeminiApiKey => write!(f, "gemini_api_key"),
            RuntimeProviderType::Anthropic => write!(f, "anthropic"),
            RuntimeProviderType::AzureOpenai => write!(f, "azure_openai"),
            RuntimeProviderType::AwsBedrock => write!(f, "aws_bedrock"),
            RuntimeProviderType::Ollama => write!(f, "ollama"),
        }
    }
}

impl FromStr for RuntimeProviderType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "openai" => Ok(Self::OpenAI),
            "claude" => Ok(Self::Claude),
            "anthropic" => Ok(Self::Anthropic),
            "anthropic_compatible" | "anthropic-compatible" => Ok(Self::AnthropicCompatible),
            "vertex" | "vertexai" | "vertex_ai" => Ok(Self::Vertex),
            "gemini_api_key" | "gemini-api-key" => Ok(Self::GeminiApiKey),
            "azure_openai" | "azure-openai" => Ok(Self::AzureOpenai),
            "aws_bedrock" | "aws-bedrock" => Ok(Self::AwsBedrock),
            "ollama" => Ok(Self::Ollama),
            "deepseek" | "deep_seek" | "deep-seek" => Ok(Self::OpenAI),
            "qwen" | "tongyi" | "dashscope" => Ok(Self::OpenAI),
            "zhipu" | "glm" | "chatglm" => Ok(Self::OpenAI),
            "moonshot" | "kimi" => Ok(Self::OpenAI),
            "baichuan" => Ok(Self::OpenAI),
            "minimax" => Ok(Self::OpenAI),
            "yi" | "01ai" => Ok(Self::OpenAI),
            "stepfun" | "step" => Ok(Self::OpenAI),
            "groq" => Ok(Self::OpenAI),
            "together" | "togetherai" => Ok(Self::OpenAI),
            "fireworks" | "fireworksai" => Ok(Self::OpenAI),
            "perplexity" => Ok(Self::OpenAI),
            "siliconflow" => Ok(Self::OpenAI),
            "oneapi" | "one-api" | "newapi" | "new-api" => Ok(Self::OpenAI),
            "custom" | "custom_openai" => Ok(Self::OpenAI),
            s if is_custom_provider_id(s) => Ok(Self::OpenAI),
            _ => Err(format!("Unknown runtime provider: {s}")),
        }
    }
}

/// Provider 声明的 Prompt Cache 模式。
///
/// 说明：
/// - 这是“上游已声明的缓存能力”，不是模型目录或协议族映射；
/// - 对普通 Provider 可为空，运行时会按 ProviderType 走默认语义；
/// - 对自定义 `anthropic-compatible` Provider，可用来覆盖默认的 `explicit_only`。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderPromptCacheMode {
    Automatic,
    ExplicitOnly,
}

/// Current API Key Provider 运行时凭证内容。
///
/// 只承载 API Key Provider 可生成的凭证形态。旧 OAuth/local CLI 变体只允许停留在
/// 历史配置反序列化与启动清理边界，不进入 current 运行时。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RuntimeCredentialData {
    OpenAIKey {
        api_key: String,
        base_url: Option<String>,
    },
    ClaudeKey {
        api_key: String,
        base_url: Option<String>,
    },
    VertexKey {
        api_key: String,
        base_url: Option<String>,
        project: String,
        location: String,
        #[serde(default)]
        model_aliases: std::collections::HashMap<String, String>,
    },
    GeminiApiKey {
        api_key: String,
        base_url: Option<String>,
        #[serde(default)]
        excluded_models: Vec<String>,
    },
    AnthropicKey {
        api_key: String,
        base_url: Option<String>,
    },
}

/// Current API Key Provider 运行时凭证 DTO。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeProviderCredential {
    pub uuid: String,
    pub provider_type: RuntimeProviderType,
    pub credential: RuntimeCredentialData,
    pub name: Option<String>,
    #[serde(default)]
    pub prompt_cache_mode_override: Option<ProviderPromptCacheMode>,
}

impl RuntimeProviderCredential {
    pub fn new(provider_type: RuntimeProviderType, credential: RuntimeCredentialData) -> Self {
        Self {
            uuid: Uuid::new_v4().to_string(),
            provider_type,
            credential,
            name: None,
            prompt_cache_mode_override: None,
        }
    }

    /// 解析当前运行时凭证应采用的 Prompt Cache 模式。
    pub fn effective_prompt_cache_mode(&self) -> Option<ProviderPromptCacheMode> {
        self.prompt_cache_mode_override.or_else(|| {
            if matches!(self.provider_type, RuntimeProviderType::AnthropicCompatible) {
                return if is_known_automatic_anthropic_compatible_host(
                    runtime_base_url(&self.credential).as_deref(),
                ) {
                    Some(ProviderPromptCacheMode::Automatic)
                } else {
                    Some(ProviderPromptCacheMode::ExplicitOnly)
                };
            }

            self.provider_type
                .supports_anthropic_prompt_cache()
                .then_some(ProviderPromptCacheMode::Automatic)
        })
    }
}

fn runtime_base_url(credential: &RuntimeCredentialData) -> Option<String> {
    match credential {
        RuntimeCredentialData::OpenAIKey { base_url, .. }
        | RuntimeCredentialData::ClaudeKey { base_url, .. }
        | RuntimeCredentialData::VertexKey { base_url, .. }
        | RuntimeCredentialData::GeminiApiKey { base_url, .. }
        | RuntimeCredentialData::AnthropicKey { base_url, .. } => base_url.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        runtime_api_key_credential_uuid, runtime_api_key_id_from_credential_uuid,
        RuntimeCredentialData, RuntimeProviderType, RUNTIME_API_KEY_CREDENTIAL_UUID_PREFIX,
    };
    use std::str::FromStr;

    #[test]
    fn runtime_provider_type_does_not_include_retired_pool_variants() {
        for retired in ["kiro", "gemini", "codex", "claude_oauth"] {
            assert!(
                RuntimeProviderType::from_str(retired).is_err(),
                "{retired} 不应进入 current RuntimeProviderType"
            );
        }
    }

    #[test]
    fn runtime_credential_data_rejects_retired_oauth_payloads() {
        for payload in [
            r#"{"type":"kiro_oauth"}"#,
            r#"{"type":"gemini_oauth"}"#,
            r#"{"type":"codex_oauth"}"#,
            r#"{"type":"claude_oauth"}"#,
        ] {
            assert!(
                serde_json::from_str::<RuntimeCredentialData>(payload).is_err(),
                "旧 OAuth payload 不应反序列化为 RuntimeCredentialData: {payload}"
            );
        }
    }

    #[test]
    fn runtime_api_key_uuid_helpers_round_trip_current_prefix() {
        let uuid = runtime_api_key_credential_uuid("key-123");

        assert_eq!(
            uuid,
            format!("{RUNTIME_API_KEY_CREDENTIAL_UUID_PREFIX}key-123")
        );
        assert_eq!(
            runtime_api_key_id_from_credential_uuid(&uuid),
            Some("key-123")
        );
        assert_eq!(
            runtime_api_key_id_from_credential_uuid(RUNTIME_API_KEY_CREDENTIAL_UUID_PREFIX),
            None
        );
        assert_eq!(
            runtime_api_key_id_from_credential_uuid("fallback-key-123"),
            None
        );
    }
}
