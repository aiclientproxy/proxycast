//! Runtime provider 配置与错误判定边界。

use app_server_protocol::ProtocolKind;

use crate::ModelProviderProtocol;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeProviderProtocol {
    ChatCompletions,
    Responses,
    AnthropicMessages,
    GeminiGenerateContent,
}

impl RuntimeProviderProtocol {
    pub fn uses_responses_api(self) -> bool {
        matches!(self, Self::Responses)
    }

    pub fn to_model_provider_protocol(self) -> ModelProviderProtocol {
        match self {
            Self::ChatCompletions => ModelProviderProtocol::ChatCompletions,
            Self::Responses => ModelProviderProtocol::Responses,
            Self::AnthropicMessages => ModelProviderProtocol::AnthropicMessages,
            Self::GeminiGenerateContent => ModelProviderProtocol::GeminiGenerateContent,
        }
    }

    pub fn from_route_protocol(protocol: &ProtocolKind) -> Option<Self> {
        match protocol {
            ProtocolKind::OpenaiChat => Some(Self::ChatCompletions),
            ProtocolKind::OpenaiResponses | ProtocolKind::CodexResponses => Some(Self::Responses),
            ProtocolKind::AnthropicMessages => Some(Self::AnthropicMessages),
            ProtocolKind::GeminiGenerateContent => Some(Self::GeminiGenerateContent),
            ProtocolKind::OpenaiImages
            | ProtocolKind::Fal
            | ProtocolKind::BedrockConverse
            | ProtocolKind::VertexGemini
            | ProtocolKind::Unknown => None,
        }
    }

    pub fn from_provider_type(provider_type: &str) -> Option<Self> {
        match normalize_provider_type(provider_type).as_str() {
            "openai" | "new_api" | "gateway" => Some(Self::ChatCompletions),
            "openai_response" | "openai_responses" | "responses" | "codex" | "ollama" => {
                Some(Self::Responses)
            }
            "anthropic" | "anthropic_compatible" => Some(Self::AnthropicMessages),
            "gemini" | "gemini_api_key" | "google" => Some(Self::GeminiGenerateContent),
            "azure" | "azure_openai" | "vertex" | "vertexai" | "vertex_ai" | "gcpvertexai"
            | "aws_bedrock" | "bedrock" | "fal" => None,
            _ => None,
        }
    }

    pub fn from_direct_route(provider_name: &str, protocol: &ProtocolKind) -> Option<Self> {
        let normalized = normalize_provider_type(provider_name);
        if matches!(normalized.as_str(), "azure" | "azure_openai") {
            return None;
        }
        if normalized == "ollama"
            && !matches!(
                protocol,
                ProtocolKind::OpenaiResponses | ProtocolKind::CodexResponses
            )
        {
            return None;
        }
        if matches!(protocol, ProtocolKind::GeminiGenerateContent)
            && !matches!(normalized.as_str(), "gemini" | "gemini_api_key" | "google")
        {
            return None;
        }
        Self::from_route_protocol(protocol)
    }
}

fn normalize_provider_type(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

/// 已解析 provider route 的认证语义。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeProviderAuth {
    ApiKey,
    NoAuth,
    OemManaged,
}

/// Runtime provider 配置。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeProviderConfig {
    /// Provider 名称 (openai, anthropic, google 等)
    pub provider_name: String,
    /// Provider 选择器（优先保留前端 provider_id / runtime_provider_type）
    pub provider_selector: Option<String>,
    /// 模型名称
    pub model_name: String,
    /// API Key
    pub api_key: Option<String>,
    /// Route resolver 明确给出的认证方式
    pub auth: RuntimeProviderAuth,
    /// Base URL
    pub base_url: Option<String>,
    /// 凭证 UUID（用于记录使用和健康状态）
    pub credential_uuid: String,
    /// 当前回合显式推理强度
    pub reasoning_effort: Option<String>,
    /// 当前回合显式服务等级
    pub service_tier: Option<String>,
    /// App Server RouteResolver 派生出的 provider 执行协议
    pub protocol: Option<RuntimeProviderProtocol>,
    /// Provider 是否显式支持 Responses WebSocket transport
    pub supports_websockets: bool,
    /// 当前回合是否启用 toolshim
    pub toolshim: bool,
    /// toolshim 解释器模型
    pub toolshim_model: Option<String>,
}

pub fn message_is_non_retryable_provider_rejection(message: &str) -> bool {
    let normalized = message.to_ascii_lowercase();
    normalized.contains("authentication error")
        || normalized.contains("unauthorized")
        || normalized.contains("forbidden")
        || !is_retryable_request_failed_message(message)
}

fn is_retryable_request_failed_message(message: &str) -> bool {
    let normalized = message.to_ascii_lowercase();
    let non_retryable_markers = [
        "bad request (400)",
        "resource not found (404)",
        "invalid_request_error",
        "status: 400",
        "status: 401",
        "status: 403",
        "status: 404",
        "status 400",
        "status 401",
        "status 403",
        "status 404",
    ];

    !non_retryable_markers
        .iter()
        .any(|marker| normalized.contains(marker))
}

#[cfg(test)]
mod tests {
    use super::{message_is_non_retryable_provider_rejection, RuntimeProviderProtocol};
    use app_server_protocol::ProtocolKind;

    #[test]
    fn route_protocol_adapter_availability_is_explicit() {
        assert_eq!(
            RuntimeProviderProtocol::from_route_protocol(&ProtocolKind::OpenaiChat),
            Some(RuntimeProviderProtocol::ChatCompletions)
        );
        assert_eq!(
            RuntimeProviderProtocol::from_route_protocol(&ProtocolKind::OpenaiResponses),
            Some(RuntimeProviderProtocol::Responses)
        );
        assert_eq!(
            RuntimeProviderProtocol::from_route_protocol(&ProtocolKind::CodexResponses),
            Some(RuntimeProviderProtocol::Responses)
        );
        assert_eq!(
            RuntimeProviderProtocol::from_route_protocol(&ProtocolKind::AnthropicMessages),
            Some(RuntimeProviderProtocol::AnthropicMessages)
        );
        assert_eq!(
            RuntimeProviderProtocol::from_route_protocol(&ProtocolKind::GeminiGenerateContent),
            Some(RuntimeProviderProtocol::GeminiGenerateContent)
        );

        for protocol in [
            ProtocolKind::OpenaiImages,
            ProtocolKind::Fal,
            ProtocolKind::BedrockConverse,
            ProtocolKind::VertexGemini,
            ProtocolKind::Unknown,
        ] {
            assert_eq!(
                RuntimeProviderProtocol::from_route_protocol(&protocol),
                None,
                "{protocol:?} must remain unavailable until its wire adapter exists"
            );
        }
    }

    #[test]
    fn provider_adapter_availability_includes_auth_transport_requirements() {
        for (provider_type, expected) in [
            ("openai", RuntimeProviderProtocol::ChatCompletions),
            ("new-api", RuntimeProviderProtocol::ChatCompletions),
            ("gateway", RuntimeProviderProtocol::ChatCompletions),
            ("openai-response", RuntimeProviderProtocol::Responses),
            ("codex", RuntimeProviderProtocol::Responses),
            ("ollama", RuntimeProviderProtocol::Responses),
            (
                "anthropic-compatible",
                RuntimeProviderProtocol::AnthropicMessages,
            ),
            ("gemini", RuntimeProviderProtocol::GeminiGenerateContent),
            (
                "gemini-api-key",
                RuntimeProviderProtocol::GeminiGenerateContent,
            ),
            ("google", RuntimeProviderProtocol::GeminiGenerateContent),
        ] {
            assert_eq!(
                RuntimeProviderProtocol::from_provider_type(provider_type),
                Some(expected),
                "provider_type={provider_type}"
            );
        }

        for provider_type in ["azure-openai", "vertexai", "aws-bedrock", "fal"] {
            assert_eq!(
                RuntimeProviderProtocol::from_provider_type(provider_type),
                None,
                "{provider_type} must remain unavailable until its full adapter exists"
            );
        }

        assert_eq!(
            RuntimeProviderProtocol::from_direct_route("azure_openai", &ProtocolKind::OpenaiChat,),
            None,
            "OpenAI-shaped bodies do not satisfy Azure auth/query requirements"
        );
        assert_eq!(
            RuntimeProviderProtocol::from_direct_route("ollama", &ProtocolKind::OpenaiResponses,),
            Some(RuntimeProviderProtocol::Responses)
        );
        assert_eq!(
            RuntimeProviderProtocol::from_direct_route("ollama", &ProtocolKind::OpenaiChat),
            None,
            "Ollama chat wire was removed; Ollama is Responses-only"
        );
        assert_eq!(
            RuntimeProviderProtocol::from_direct_route(
                "google",
                &ProtocolKind::GeminiGenerateContent,
            ),
            Some(RuntimeProviderProtocol::GeminiGenerateContent)
        );
        assert_eq!(
            RuntimeProviderProtocol::from_direct_route(
                "openai",
                &ProtocolKind::GeminiGenerateContent,
            ),
            None,
            "Gemini wire must not be admitted under an unrelated direct provider identity"
        );
    }

    #[test]
    fn classifies_non_retryable_provider_rejections() {
        assert!(message_is_non_retryable_provider_rejection(
            "Request failed: Bad request (400): 当前模型未在租户白名单中开放"
        ));
        assert!(message_is_non_retryable_provider_rejection(
            "Authentication error: invalid key"
        ));
        assert!(!message_is_non_retryable_provider_rejection(
            "connection failed"
        ));
        assert!(!message_is_non_retryable_provider_rejection(
            "Server error: temporarily unavailable"
        ));
    }
}
