//! Provider 类型定义
//!
//! 包含 Provider 类型枚举和相关实现。

use serde::{Deserialize, Serialize};

/// 是否为自定义 Provider ID（`custom-*`）
pub fn is_custom_provider_id(provider_type: &str) -> bool {
    provider_type.to_lowercase().starts_with("custom-")
}

/// Provider 类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    Kiro,
    Gemini,
    #[serde(rename = "openai")]
    OpenAI,
    Claude,
    #[serde(rename = "claude_oauth")]
    ClaudeOAuth,
    /// Anthropic 兼容格式（支持 system 数组格式等变体）
    #[serde(rename = "anthropic_compatible")]
    AnthropicCompatible,
    Vertex,
    #[serde(rename = "gemini_api_key")]
    GeminiApiKey,
    Codex,
    // API Key Provider 类型
    Anthropic,
    #[serde(rename = "azure_openai")]
    AzureOpenai,
    #[serde(rename = "aws_bedrock")]
    AwsBedrock,
    Ollama,
}

impl ProviderType {
    /// 是否支持自动注入 Anthropic prompt cache 标记。
    ///
    /// `AnthropicCompatible` 只表示请求协议兼容，不能等同于上游具备 Anthropic prompt caching 能力。
    pub const fn supports_anthropic_prompt_cache(&self) -> bool {
        matches!(self, Self::Claude | Self::ClaudeOAuth | Self::Anthropic)
    }
}

impl std::fmt::Display for ProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderType::Kiro => write!(f, "kiro"),
            ProviderType::Gemini => write!(f, "gemini"),
            ProviderType::OpenAI => write!(f, "openai"),
            ProviderType::Claude => write!(f, "claude"),
            ProviderType::ClaudeOAuth => write!(f, "claude_oauth"),
            ProviderType::AnthropicCompatible => write!(f, "anthropic_compatible"),
            ProviderType::Vertex => write!(f, "vertex"),
            ProviderType::GeminiApiKey => write!(f, "gemini_api_key"),
            ProviderType::Codex => write!(f, "codex"),
            ProviderType::Anthropic => write!(f, "anthropic"),
            ProviderType::AzureOpenai => write!(f, "azure_openai"),
            ProviderType::AwsBedrock => write!(f, "aws_bedrock"),
            ProviderType::Ollama => write!(f, "ollama"),
        }
    }
}

impl std::str::FromStr for ProviderType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "kiro" => Ok(ProviderType::Kiro),
            "gemini" => Ok(ProviderType::Gemini),
            "openai" => Ok(ProviderType::OpenAI),
            "claude" => Ok(ProviderType::Claude),
            "claude_oauth" => Ok(ProviderType::ClaudeOAuth),
            "anthropic_compatible" | "anthropic-compatible" => {
                Ok(ProviderType::AnthropicCompatible)
            }
            "vertex" => Ok(ProviderType::Vertex),
            "gemini_api_key" => Ok(ProviderType::GeminiApiKey),
            "codex" => Ok(ProviderType::Codex),
            "anthropic" => Ok(ProviderType::Anthropic),
            "azure_openai" | "azure-openai" => Ok(ProviderType::AzureOpenai),
            "aws_bedrock" | "aws-bedrock" => Ok(ProviderType::AwsBedrock),
            "ollama" => Ok(ProviderType::Ollama),
            // OpenAI 兼容的第三方 Provider 映射到 OpenAI
            "deepseek" | "deep_seek" | "deep-seek" => Ok(ProviderType::OpenAI),
            "qwen" | "tongyi" | "dashscope" => Ok(ProviderType::OpenAI),
            "zhipu" | "glm" | "chatglm" => Ok(ProviderType::OpenAI),
            "moonshot" | "kimi" => Ok(ProviderType::OpenAI),
            "baichuan" => Ok(ProviderType::OpenAI),
            "minimax" => Ok(ProviderType::OpenAI),
            "yi" | "01ai" => Ok(ProviderType::OpenAI),
            "stepfun" | "step" => Ok(ProviderType::OpenAI),
            "groq" => Ok(ProviderType::OpenAI),
            "together" | "togetherai" => Ok(ProviderType::OpenAI),
            "fireworks" | "fireworksai" => Ok(ProviderType::OpenAI),
            "perplexity" => Ok(ProviderType::OpenAI),
            "nearai" | "near-ai" | "near_ai" => Ok(ProviderType::OpenAI),
            "siliconflow" => Ok(ProviderType::OpenAI),
            "oneapi" | "one-api" | "newapi" | "new-api" => Ok(ProviderType::OpenAI),
            "custom" | "custom_openai" => Ok(ProviderType::OpenAI),
            // 自定义 Provider（UUID 格式，如 custom-ba4e7574-dd00-4784-945a-0f383dfa1272）
            // 注意：这里仅做“通道级”兼容映射（按 OpenAI 兜底），
            // 实际协议应在运行时通过 API Key Provider.type 决定。
            s if is_custom_provider_id(s) => Ok(ProviderType::OpenAI),
            _ => Err(format!("Unknown provider: {s}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_type_from_str() {
        assert_eq!("kiro".parse::<ProviderType>().unwrap(), ProviderType::Kiro);
        assert_eq!(
            "gemini".parse::<ProviderType>().unwrap(),
            ProviderType::Gemini
        );
        assert_eq!(
            "openai".parse::<ProviderType>().unwrap(),
            ProviderType::OpenAI
        );
        assert_eq!(
            "claude".parse::<ProviderType>().unwrap(),
            ProviderType::Claude
        );
        assert_eq!(
            "vertex".parse::<ProviderType>().unwrap(),
            ProviderType::Vertex
        );
        assert_eq!(
            "gemini_api_key".parse::<ProviderType>().unwrap(),
            ProviderType::GeminiApiKey
        );
        assert_eq!("KIRO".parse::<ProviderType>().unwrap(), ProviderType::Kiro);
        assert_eq!(
            "Gemini".parse::<ProviderType>().unwrap(),
            ProviderType::Gemini
        );
        assert_eq!(
            "VERTEX".parse::<ProviderType>().unwrap(),
            ProviderType::Vertex
        );
        assert!("invalid".parse::<ProviderType>().is_err());
    }

    #[test]
    fn test_custom_provider_uuid_format() {
        // 自定义 Provider UUID 格式应该映射到 OpenAI
        assert_eq!(
            "custom-ba4e7574-dd00-4784-945a-0f383dfa1272"
                .parse::<ProviderType>()
                .unwrap(),
            ProviderType::OpenAI
        );
        assert_eq!(
            "custom-12345678-1234-1234-1234-123456789abc"
                .parse::<ProviderType>()
                .unwrap(),
            ProviderType::OpenAI
        );
        // 普通 custom 也应该映射到 OpenAI
        assert_eq!(
            "custom".parse::<ProviderType>().unwrap(),
            ProviderType::OpenAI
        );
    }

    #[test]
    fn test_is_custom_provider_id() {
        assert!(is_custom_provider_id(
            "custom-ba4e7574-dd00-4784-945a-0f383dfa1272"
        ));
        assert!(is_custom_provider_id(
            "CUSTOM-ba4e7574-dd00-4784-945a-0f383dfa1272"
        ));
        assert!(!is_custom_provider_id("custom"));
        assert!(!is_custom_provider_id("openai"));
    }

    #[test]
    fn test_provider_type_display() {
        assert_eq!(ProviderType::Kiro.to_string(), "kiro");
        assert_eq!(ProviderType::Gemini.to_string(), "gemini");
        assert_eq!(ProviderType::OpenAI.to_string(), "openai");
        assert_eq!(ProviderType::Claude.to_string(), "claude");
        assert_eq!(ProviderType::Vertex.to_string(), "vertex");
        assert_eq!(ProviderType::GeminiApiKey.to_string(), "gemini_api_key");
    }

    #[test]
    fn test_provider_type_serde() {
        assert_eq!(
            serde_json::to_string(&ProviderType::Kiro).unwrap(),
            "\"kiro\""
        );
        assert_eq!(
            serde_json::to_string(&ProviderType::OpenAI).unwrap(),
            "\"openai\""
        );
        assert_eq!(
            serde_json::from_str::<ProviderType>("\"kiro\"").unwrap(),
            ProviderType::Kiro
        );
        assert_eq!(
            serde_json::from_str::<ProviderType>("\"openai\"").unwrap(),
            ProviderType::OpenAI
        );
    }

    #[test]
    fn test_supports_anthropic_prompt_cache() {
        assert!(ProviderType::Claude.supports_anthropic_prompt_cache());
        assert!(ProviderType::Anthropic.supports_anthropic_prompt_cache());
        assert!(!ProviderType::AnthropicCompatible.supports_anthropic_prompt_cache());
        assert!(!ProviderType::OpenAI.supports_anthropic_prompt_cache());
    }
}
